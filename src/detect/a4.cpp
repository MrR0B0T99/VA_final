#include "detect/a4.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace detect {

bool orderFourCorners(const std::vector<cv::Point>& approx,
                      std::vector<cv::Point2f>& ordered) {
  if (approx.size() != 4) return false;

  std::vector<cv::Point2f> pts;
  pts.reserve(4);
  for (const cv::Point& p : approx) pts.emplace_back(p);

  std::sort(pts.begin(), pts.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
    if (std::abs(a.y - b.y) > 1e-3f) return a.y < b.y;
    return a.x < b.x;
  });

  std::vector<cv::Point2f> top(pts.begin(), pts.begin() + 2);
  std::vector<cv::Point2f> bottom(pts.begin() + 2, pts.end());

  std::sort(top.begin(), top.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
    return a.x < b.x;
  });
  std::sort(bottom.begin(), bottom.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
    return a.x < b.x;
  });

  // TL, BL, BR, TR (dans l'espace image)
  ordered = {top[0], bottom[0], bottom[1], top[1]};

  const double EPS = 1e-3;
  for (int i = 0; i < 4; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      if (cv::norm(ordered[i] - ordered[j]) < EPS) {
        ordered.clear();
        return false; // points non distincts
      }
    }
  }

  return true;
}

bool detectA4Corners(const cv::Mat& frameBGR,
                     std::vector<cv::Point2f>& imagePts) {
  imagePts.clear();
  if (frameBGR.empty()) return false;

  cv::Mat gray; cv::cvtColor(frameBGR, gray, cv::COLOR_BGR2GRAY);

  cv::Mat eq;
  {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(gray, eq);
  }

  cv::Mat blurred; cv::GaussianBlur(eq, blurred, cv::Size(5,5), 0);

  cv::Mat adaptive;
  cv::adaptiveThreshold(blurred, adaptive, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv::THRESH_BINARY_INV, 31, 5);

  cv::Mat edges; cv::Canny(blurred, edges, 35, 105, 3, true);

  cv::Mat combined; cv::bitwise_or(adaptive, edges, combined);
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
  cv::morphologyEx(combined, combined, cv::MORPH_CLOSE, kernel, cv::Point(-1,-1), 2);
  cv::dilate(combined, combined, kernel, cv::Point(-1,-1), 1);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(combined, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) return false;

  const double imgArea = static_cast<double>(frameBGR.cols * frameBGR.rows);
  const double targetRatio = 297.0 / 210.0; // ~1.414

  double bestScore = -1.0;
  std::vector<cv::Point> bestApprox;

  for (const auto& contour : contours) {
    double contourAreaVal = std::fabs(cv::contourArea(contour));
    if (contourAreaVal < 0.0005 * imgArea) continue; // ignore trop petits éléments

    double peri = cv::arcLength(contour, true);
    std::vector<cv::Point> approx;
    cv::approxPolyDP(contour, approx, 0.02 * peri, true);
    if (approx.size() != 4) continue;
    if (!cv::isContourConvex(approx)) continue;

    double approxArea = std::fabs(cv::contourArea(approx));
    if (approxArea < 0.0005 * imgArea) continue;

    std::vector<cv::Point2f> orderedCandidate;
    if (!orderFourCorners(approx, orderedCandidate)) continue;

    cv::RotatedRect box = cv::minAreaRect(orderedCandidate);
    double widthTop    = cv::norm(orderedCandidate[1] - orderedCandidate[0]);
    double widthBottom = cv::norm(orderedCandidate[3] - orderedCandidate[2]);
    double heightLeft  = cv::norm(orderedCandidate[2] - orderedCandidate[0]);
    double heightRight = cv::norm(orderedCandidate[3] - orderedCandidate[1]);

    double widthAvg  = (widthTop + widthBottom) * 0.5;
    double heightAvg = (heightLeft + heightRight) * 0.5;
    if (widthAvg < 1.0 || heightAvg < 1.0) continue;

    cv::Size2f boxSize = box.size;
    double boxWidth = std::max(boxSize.width, boxSize.height);
    double boxHeight = std::min(boxSize.width, boxSize.height);
    double ratio = (boxHeight > 0.0) ? (boxWidth / boxHeight) : 0.0;
    if (ratio < 1.05 || ratio > 1.9) continue;

    double boundingArea = widthAvg * heightAvg;
    double solidity = approxArea / boundingArea;
    if (solidity < 0.7) continue;

    double score = approxArea - std::abs(ratio - targetRatio) * 500.0;
    score -= std::abs(widthTop - widthBottom) * 2.0;
    score -= std::abs(heightLeft - heightRight) * 2.0;
    if (score > bestScore) {
      bestScore = score;
      bestApprox = approx;
    }
  }

  if (bestApprox.size() != 4) return false;

  std::vector<cv::Point2f> ordered;
  if (!orderFourCorners(bestApprox, ordered)) return false;

  cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01);
  cv::cornerSubPix(gray, ordered, cv::Size(5,5), cv::Size(-1,-1), criteria);

  imagePts = std::move(ordered);
  return true;
}

void drawOrderedCorners(cv::Mat& img, const std::vector<cv::Point2f>& pts) {
  if (pts.size()!=4) return;

  // TL, BL, BR, TR
  const cv::Scalar colors[4] = {
    {0,0,255},   // TL red
    {0,255,255}, // BL yellow
    {255,0,0},   // BR blue
    {0,255,0}    // TR green
  };
  const char* labels[4] = {"TL","BL","BR","TR"};

  for (int i=0;i<4;i++){
    cv::circle(img, pts[i], 8, colors[i], -1, cv::LINE_AA);
    cv::putText(img, labels[i], pts[i] + cv::Point2f(10,-10),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2, cv::LINE_AA);
  }
}

} // namespace detect
