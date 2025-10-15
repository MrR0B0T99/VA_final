#pragma once
#include <opencv2/core.hpp>
#include <vector>

/**
 * @file a4.hpp
 * @brief Détection d'une feuille A4 (contour quadrilatère) et ordonnancement des coins.
 */
namespace detect {

/**
 * @brief Ordonne 4 points en (TL, BL, BR, TR) selon l'image.
 * @param approx 4 points issus d'une approx polygone.
 * @param[out] ordered Points float ordonnés (taille 4).
 * @return true si succès.
 */
bool orderFourCorners(const std::vector<cv::Point>& approx,
                      std::vector<cv::Point2f>& ordered);

/**
 * @brief Détecte les quatre coins d'une feuille A4 sur l'image.
 * @param frameBGR Image d'entrée (BGR)
 * @param[out] imagePts Points image ordonnés (TL,BL,BR,TR)
 * @return true si détecté et ordonné
 */
bool detectA4Corners(const cv::Mat& frameBGR,
                     std::vector<cv::Point2f>& imagePts);

/**
 * @brief Dessine les 4 coins ordonnés avec labels (TL/BL/BR/TR) pour debug.
 */
void drawOrderedCorners(cv::Mat& img, const std::vector<cv::Point2f>& pts);

} // namespace detect
