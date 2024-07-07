cmake_minimum_required(VERSION 3.10)

# Название проекта
project(SundialsExample CXX)

# Задаем стандарт C++
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Указываем путь к SUNDIALS, если необходимо
set(SUNDIALS_DIR /usr/local)

# Добавляем путь к модулям CMake
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${SUNDIALS_DIR}/lib/cmake/sundials")

# Устанавливаем необходимые пути для поиска библиотек
find_package(SUNDIALS REQUIRED)

# Добавляем исходный файл
add_executable(sundial src/sundial.cpp)

# Линкуем библиотеки SUNDIALS к исполняемому файлу
target_link_libraries(sundial PRIVATE sundials_cvode sundials_nvecserial sundials_sunlinsol sundials_sunmatrix)
