#include "detect/a4.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

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

  const cv::Point2f& TL = top[0];
  const cv::Point2f& TR = top[1];
  const cv::Point2f& BL = bottom[0];
  const cv::Point2f& BR = bottom[1];

  ordered = {TL, BL, BR, TR};

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

  cv::Mat hsv; cv::cvtColor(frameBGR, hsv, cv::COLOR_BGR2HSV);
  cv::Mat whiteMask;
  const int S_MAX = 90;
  const int V_MIN = 150;
  cv::inRange(hsv, cv::Scalar(0, 0, V_MIN), cv::Scalar(179, S_MAX, 255), whiteMask);

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
  cv::morphologyEx(whiteMask, whiteMask, cv::MORPH_CLOSE, kernel, cv::Point(-1,-1), 2);
  cv::morphologyEx(whiteMask, whiteMask, cv::MORPH_OPEN, kernel, cv::Point(-1,-1), 1);

  cv::Mat combined; cv::bitwise_or(adaptive, edges, combined);
  cv::bitwise_and(combined, whiteMask, combined);
  cv::morphologyEx(combined, combined, cv::MORPH_CLOSE, kernel, cv::Point(-1,-1), 2);
  cv::dilate(combined, combined, kernel, cv::Point(-1,-1), 1);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(combined, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) return false;

  const double imgArea = static_cast<double>(frameBGR.cols * frameBGR.rows);
  const double targetRatio = 297.0 / 210.0; // ~1.414
  const double minAreaRatio = 0.0005;

  double bestScore = -1.0;
  std::vector<cv::Point> bestApprox;

  for (const auto& contour : contours) {
    double contourAreaVal = std::fabs(cv::contourArea(contour));
    if (contourAreaVal < minAreaRatio * imgArea) continue; // ignore trop petits éléments

    double peri = cv::arcLength(contour, true);
    std::vector<cv::Point> approx;
    cv::approxPolyDP(contour, approx, 0.02 * peri, true);
    if (approx.size() != 4) continue;
    if (!cv::isContourConvex(approx)) continue;

    double approxArea = std::fabs(cv::contourArea(approx));
    if (approxArea < minAreaRatio * imgArea) continue;

    std::vector<cv::Point2f> orderedCandidate;
    if (!orderFourCorners(approx, orderedCandidate)) continue;

    cv::RotatedRect box = cv::minAreaRect(orderedCandidate);
    double topWidth    = cv::norm(orderedCandidate[3] - orderedCandidate[0]);
    double bottomWidth = cv::norm(orderedCandidate[2] - orderedCandidate[1]);
    double leftHeight  = cv::norm(orderedCandidate[1] - orderedCandidate[0]);
    double rightHeight = cv::norm(orderedCandidate[2] - orderedCandidate[3]);

    double widthAvg  = (topWidth + bottomWidth) * 0.5;
    double heightAvg = (leftHeight + rightHeight) * 0.5;
    if (widthAvg < 1.0 || heightAvg < 1.0) continue;

    cv::Size2f boxSize = box.size;
    double boxWidth = std::max(boxSize.width, boxSize.height);
    double boxHeight = std::min(boxSize.width, boxSize.height);
    double ratio = (boxHeight > 0.0) ? (boxWidth / boxHeight) : 0.0;
    if (ratio < 1.05 || ratio > 1.9) continue;

    auto angleDeg = [](const cv::Point2f& prev, const cv::Point2f& center,
                       const cv::Point2f& next) {
      cv::Point2f v1 = prev - center;
      cv::Point2f v2 = next - center;
      double denom = cv::norm(v1) * cv::norm(v2);
      if (denom <= std::numeric_limits<double>::epsilon()) return 180.0;
      double cosTheta = (v1.dot(v2)) / denom;
      cosTheta = std::max(-1.0, std::min(1.0, cosTheta));
      return std::acos(cosTheta) * 180.0 / CV_PI;
    };

    double anglePenalty = 0.0;
    bool validAngles = true;
    for (int i = 0; i < 4 && validAngles; ++i) {
      const cv::Point2f& prev = orderedCandidate[(i + 3) % 4];
      const cv::Point2f& center = orderedCandidate[i];
      const cv::Point2f& next = orderedCandidate[(i + 1) % 4];
      double angle = angleDeg(prev, center, next);
      if (angle < 55.0 || angle > 125.0) {
        validAngles = false;
      } else {
        anglePenalty += std::abs(angle - 90.0);
      }
    }
    if (!validAngles) continue;

    double boundingArea = widthAvg * heightAvg;
    double solidity = approxArea / boundingArea;
    if (solidity < 0.7) continue;

    std::vector<cv::Point> polyInt;
    polyInt.reserve(4);
    for (const auto& p : orderedCandidate) {
      polyInt.emplace_back(cvRound(p.x), cvRound(p.y));
    }
    cv::Mat candidateMask = cv::Mat::zeros(whiteMask.size(), CV_8U);
    cv::fillConvexPoly(candidateMask, polyInt, 255, cv::LINE_AA);

    double whiteMean = cv::mean(whiteMask, candidateMask).val[0] / 255.0;
    whiteMean = std::clamp(whiteMean, 0.0, 1.0);
    if (whiteMean < 0.2) continue;

    cv::Mat structureMask;
    cv::bitwise_and(combined, candidateMask, structureMask);
    double structureRatio = approxArea > 0.0
                               ? static_cast<double>(cv::countNonZero(structureMask)) /
                                     std::max(approxArea, 1.0)
                               : 0.0;
    structureRatio = std::clamp(structureRatio, 0.0, 1.0);

    double score = approxArea;
    score -= std::abs(ratio - targetRatio) * 500.0;
    score -= (std::abs(topWidth - bottomWidth) + std::abs(leftHeight - rightHeight)) * 2.0;
    score -= anglePenalty * 10.0;
    score += whiteMean * 2000.0;
    score += structureRatio * 1500.0;
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

  std::vector<cv::Point> poly;
  poly.reserve(4);
  for (const cv::Point2f& p : pts) {
    poly.emplace_back(cvRound(p.x), cvRound(p.y));
  }
  cv::polylines(img, poly, true, cv::Scalar(0, 220, 0), 2, cv::LINE_AA);

  for (int i=0;i<4;i++){
    cv::Point center(cvRound(pts[i].x), cvRound(pts[i].y));
    cv::circle(img, center, 8, colors[i], -1, cv::LINE_AA);

    std::ostringstream oss;
    oss << labels[i] << " (" << center.x << ", " << center.y << ")";
    cv::Point textPos(center.x + 10, center.y - 10);
    cv::putText(img, oss.str(), textPos, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                colors[i], 2, cv::LINE_AA);
  }
}

} // namespace detect
