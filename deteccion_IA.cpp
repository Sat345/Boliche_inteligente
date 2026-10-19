//
// Created by sait1 on 19/10/2026.
//

#include "deteccion_IA.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>


const float CONFIDENCE_THRESHOLD = 0.5f;
const float NMS_THRESHOLD = 0.4f;

void procesarVideo() {

    // Cargar el modelo YOLOv11 entrenado en formato ONNX
    cv::dnn::Net net = cv::dnn::readNetFromONNX("bolos.onnx");

    // Optimizaciones estándar para CPU
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Conectar a la cámara (0 para webcam USB, cambia por ip_celular si usas el teléfono)
    std::string ip_celular = "http://192.168.137.229:8080/video";
    cv::VideoCapture cam(0);

    if (!cam.isOpened()) {
        std::cout << "Error: No se pudo conectar a la camara. Verifica la IP." << std::endl;
        return;
    }

    cv::Mat frame;
    int contador_cuadros = 0;

    // Declaración única de vectores persistentes
    static std::vector<cv::Rect> b_boxes;
    static std::vector<float> b_confidences;
    static std::vector<int> b_indices;
    static std::vector<int> b_class_ids;

    while (cam.read(frame)) {
        contador_cuadros++;

        // Saltamos cuadros para acelerar el procesamiento en tu CPU
        bool procesar_IA = (contador_cuadros % 4 == 0);

        int width = frame.cols;
        int height = frame.rows;

        if (procesar_IA) {
            // Limpieza de todos los vectores antes de reescribir
            b_boxes.clear();
            b_confidences.clear();
            b_indices.clear();
            b_class_ids.clear();

            // Preparar la imagen para YOLO (640x640)
            cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
            net.setInput(blob);

            // Ejecutar la detección
            std::vector<cv::Mat> outputs;
            net.forward(outputs, net.getUnconnectedOutLayersNames());

            // Decodificar salida de YOLOv11 Multi-Clase
            float* data = (float*)outputs[0].data;

            for (int i = 0; i < 8400; ++i) {
                float score_bola = data[4 * 8400 + i];  // Clase 0
                float score_bolo = data[5 * 8400 + i];  // Clase 1
                float score_barra = data[6 * 8400 + i]; // Clase 2

                // Encontrar el valor más alto entre los tres
                float confidence = score_bolo;
                int class_id = 1; // Por defecto es bolo

                if (score_bola > confidence) {
                    confidence = score_bola;
                    class_id = 0;
                }
                if (score_barra > confidence) {
                    confidence = score_barra;
                    class_id = 2;
                }

                // Filtrar por el umbral de confianza
                if (confidence >= CONFIDENCE_THRESHOLD) {
                    float cx = data[0 * 8400 + i] * (width / 640.0f);
                    float cy = data[1 * 8400 + i] * (height / 640.0f);
                    float w = data[2 * 8400 + i] * (width / 640.0f);
                    float h = data[3 * 8400 + i] * (height / 640.0f);

                    int x = static_cast<int>(cx - w / 2);
                    int y = static_cast<int>(cy - h / 2);

                    b_boxes.push_back(cv::Rect(x, y, static_cast<int>(w), static_cast<int>(h)));
                    b_confidences.push_back(confidence);
                    b_class_ids.push_back(class_id);
                }
            }

            // ¡ESTA LÍNEA ES LA QUE FALTABA! (Aplica el filtro NMS y llena b_indices)
            cv::dnn::NMSBoxes(b_boxes, b_confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, b_indices);
        }

        int bolos_de_pie = 0;
        bool barra_detectada = false;

        // Bucle de dibujo
        for (int idx : b_indices) {
            if (idx >= b_boxes.size() || idx >= b_class_ids.size()) continue;

            cv::Rect box = b_boxes[idx];
            int clase = b_class_ids[idx];

            if (clase == 1) { // Bolo
                bolos_de_pie++;
                cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2); // Verde
                std::string label = "Bolo: " + std::to_string(int(b_confidences[idx] * 100)) + "%";
                cv::putText(frame, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }
            else if (clase == 0) { // Bola
                cv::rectangle(frame, box, cv::Scalar(255, 0, 0), 2); // Azul
                std::string label = "Bola: " + std::to_string(int(b_confidences[idx] * 100)) + "%";
                cv::putText(frame, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
            }
            else if (clase == 2) { // Barra barredora
                barra_detectada = true;
                cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 2); // Rojo
                std::string label = "Barra Barredora: " + std::to_string(int(b_confidences[idx] * 100)) + "%";
                cv::putText(frame, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
            }
        }

        // Pintar el marcador en la pantalla original
        int bolos_caidos = 10 - bolos_de_pie;
        if (bolos_caidos < 0) bolos_caidos = 0;

        cv::putText(frame, "Bolos de pie: " + std::to_string(bolos_de_pie), cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        cv::putText(frame, "Puntaje Tiro: " + std::to_string(bolos_caidos), cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);

        // Redimensionar la ventana para optimizar el rendimiento visual
        cv::Mat frame_reducido;
        cv::resize(frame, frame_reducido, cv::Size(800, 450));

        cv::imshow("Sistema de Boliche con IA", frame_reducido);

        if (cv::waitKey(1) == 27) break; // ESC para salir
    }

    cam.release();
    cv::destroyAllWindows();
}