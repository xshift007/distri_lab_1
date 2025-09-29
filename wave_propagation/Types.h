#pragma once  // Esto es para que se compile una sola vez y no ocurran errores

enum class ScheduleType {      //Enumeracion con Scope para evitar colisiones de nombre
    Static,         // opciones del planificador
    Dynamic,
    Guided
};
