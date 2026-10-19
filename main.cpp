#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>

// Umbrales de confianza para la IA
const float CONFIDENCE_THRESHOLD = 0.5f;
const float NMS_THRESHOLD = 0.4f;

int main() {
    //Cargar el modelo YOLOv11 entrenado en formato ONNX
    cv::dnn::Net net = cv::dnn::readNetFromONNX("bolos.onnx");

    //Optimizaciones de velocidad para CPU/GPU de Windows
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    //Optimizaciones de velocidad para una computadora con tarjeta de video NVIDIA
    //net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    //net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);


    //Conectar a la cámara de tu celular "http://192.168.137.229:8080/video"
    std::string ip_celular = "http://192.168.137.229:8080/video";
    cv::VideoCapture cam(ip_celular);

    if (!cam.isOpened()) {
        std::cout << "Error: No se pudo conectar a la camara. Verifica la IP." << std::endl;
        return -1;
    }

    cv::Mat frame;
    int contador_cuadros = 0;

    while (cam.read(frame)) {
        contador_cuadros++;

        //Saltamos cuadros para acelerar la velocidad del video.
        bool procesar_IA = (contador_cuadros % 4 == 0);

        int width = frame.cols;
        int height = frame.rows;

        //Listas persistentes para mantener los dibujos en pantalla durante los cuadros que nos saltamos
        static std::vector<cv::Rect> b_boxes;
        static std::vector<float> b_confidences;
        static std::vector<int> b_indices;

        if (procesar_IA) {
            //Limpiamos los datos anteriores
            b_boxes.clear();
            b_confidences.clear();
            b_indices.clear();

            //Preparar la imagen para YOLO
            cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
            net.setInput(blob);

            //Ejecutar la detección
            std::vector<cv::Mat> outputs;
            net.forward(outputs, net.getUnconnectedOutLayersNames());

            //Decodificar salida de YOLOv11
            float* data = (float*)outputs[0].data;

            for (int i = 0; i < 8400; ++i) {
                float confidence = data[4 * 8400 + i];
                if (confidence >= CONFIDENCE_THRESHOLD) {
                    float cx = data[0 * 8400 + i] * (width / 640.0f);
                    float cy = data[1 * 8400 + i] * (height / 640.0f);
                    float w = data[2 * 8400 + i] * (width / 640.0f);
                    float h = data[3 * 8400 + i] * (height / 640.0f);

                    int x = static_cast<int>(cx - w / 2);
                    int y = static_cast<int>(cy - h / 2);

                    b_boxes.push_back(cv::Rect(x, y, static_cast<int>(w), static_cast<int>(h)));
                    b_confidences.push_back(confidence);
                }
            }
            //NMS
            cv::dnn::NMSBoxes(b_boxes, b_confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, b_indices);
        }

        //Dibujar los resultados
        int bolos_de_pie = 0;
        for (int idx : b_indices) {
            bolos_de_pie++;
            cv::Rect box = b_boxes[idx];
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
            std::string label = "Bolo: " + std::to_string(int(b_confidences[idx] * 100)) + "%";
            cv::putText(frame, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }

        int bolos_caidos = 10 - bolos_de_pie;
        cv::putText(frame, "Bolos de pie: " + std::to_string(bolos_de_pie), cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        cv::putText(frame, "Puntaje Tiro: " + std::to_string(bolos_caidos), cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);

        //Reducir la resolución de la ventana de visualización en pantalla
        // Esto disminuye drásticamente el trabajo de la tarjeta integrada Intel UHD
        cv::Mat frame_reducido;
        cv::resize(frame, frame_reducido, cv::Size(800, 450));

        cv::imshow("Sistema de Boliche con IA", frame_reducido);

        if (cv::waitKey(1) == 27) break;
    }


    cam.release();
    cv::destroyAllWindows();
    return 0;
}