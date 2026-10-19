cmake_minimum_required(VERSION 3.20)
project(Boliche_inteligente)

set(CMAKE_CXX_STANDARD 17)

# Indicamos la ruta de tu OpenCV actual
set(OpenCV_DIR "C:/ProgramData/opencv/build")

# Buscador automático de CMake
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Boliche_inteligente main.cpp)

# Enlazamos automáticamente todas las librerías
target_link_libraries(Boliche_inteligente ${OpenCV_LIBS})
