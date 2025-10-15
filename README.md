# AR A4 Demo

Cette démo montre un petit pipeline de réalité augmentée :

1. Acquisition vidéo (fichier MP4 par défaut ou webcam en direct).
2. Détection robuste des 4 coins d'une feuille A4 en conservant l'ordre **TL → BL → BR → TR**.
3. Estimation de pose via `solvePnP`.
4. Rendu OpenGL d'un cube et des axes alignés sur la feuille.

## Compilation

```bash
cmake -S . -B build
cmake --build build
```

> **Remarque** : OpenCV (modules core, imgproc, videoio, calib3d, highgui) et OpenGL/GLFW/GLEW doivent être installés.

## Exécution

```bash
./build/AR_A4_Video
```

La scène démarre sur la vidéo de démonstration (`data/Video_AR_1.mp4`).

### Basculer entre MP4 et webcam

- `V` : force la lecture du MP4.
- `W` : bascule sur la webcam (`cv::VideoCapture(0)`).

Si la webcam n'est pas disponible, un message d'erreur est affiché et la source courante est conservée.

Un message dans la console rappelle en permanence la source active et les touches.

## Conseils pour tester la détection A4

- Assurez-vous que la feuille est bien éclairée mais sans reflets directs.
- Gardez autant que possible les coins visibles dans l'image.
- La pré-contraste adaptatif (CLAHE + seuillage adaptatif + Canny fusionné) aide à gérer les variations d'éclairage et de distance.
- Les coins raffinés (cornerSubPix) donnent des poses plus stables et un rendu temps réel fluide.

## Répertoire des données

Les ressources (vidéo d'exemple, calibration) se trouvent dans `data/`.

- `Video_AR_1.mp4` : séquence démo.
- `camera.yaml` : paramètres intrinsèques utilisés au runtime.
