// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from my_interfaces:msg/AircraftData.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "my_interfaces/msg/detail/aircraft_data__struct.h"
#include "my_interfaces/msg/detail/aircraft_data__functions.h"

ROSIDL_GENERATOR_C_IMPORT
bool std_msgs__msg__header__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * std_msgs__msg__header__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool my_interfaces__msg__aircraft_data__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[46];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("my_interfaces.msg._aircraft_data.AircraftData", full_classname_dest, 45) == 0);
  }
  my_interfaces__msg__AircraftData * ros_message = _ros_message;
  {  // header
    PyObject * field = PyObject_GetAttrString(_pymsg, "header");
    if (!field) {
      return false;
    }
    if (!std_msgs__msg__header__convert_from_py(field, &ros_message->header)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // north
    PyObject * field = PyObject_GetAttrString(_pymsg, "north");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->north = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // east
    PyObject * field = PyObject_GetAttrString(_pymsg, "east");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->east = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // down
    PyObject * field = PyObject_GetAttrString(_pymsg, "down");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->down = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // altitude
    PyObject * field = PyObject_GetAttrString(_pymsg, "altitude");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->altitude = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // roll
    PyObject * field = PyObject_GetAttrString(_pymsg, "roll");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->roll = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // pitch
    PyObject * field = PyObject_GetAttrString(_pymsg, "pitch");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->pitch = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // yaw
    PyObject * field = PyObject_GetAttrString(_pymsg, "yaw");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->yaw = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // roll_deg
    PyObject * field = PyObject_GetAttrString(_pymsg, "roll_deg");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->roll_deg = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // pitch_deg
    PyObject * field = PyObject_GetAttrString(_pymsg, "pitch_deg");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->pitch_deg = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // yaw_deg
    PyObject * field = PyObject_GetAttrString(_pymsg, "yaw_deg");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->yaw_deg = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // u_forward
    PyObject * field = PyObject_GetAttrString(_pymsg, "u_forward");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->u_forward = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // v_sideways
    PyObject * field = PyObject_GetAttrString(_pymsg, "v_sideways");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->v_sideways = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // w_downward
    PyObject * field = PyObject_GetAttrString(_pymsg, "w_downward");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->w_downward = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // airspeed
    PyObject * field = PyObject_GetAttrString(_pymsg, "airspeed");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->airspeed = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // v_north
    PyObject * field = PyObject_GetAttrString(_pymsg, "v_north");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->v_north = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // v_east
    PyObject * field = PyObject_GetAttrString(_pymsg, "v_east");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->v_east = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // v_down
    PyObject * field = PyObject_GetAttrString(_pymsg, "v_down");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->v_down = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // climb_rate
    PyObject * field = PyObject_GetAttrString(_pymsg, "climb_rate");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->climb_rate = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // ground_speed
    PyObject * field = PyObject_GetAttrString(_pymsg, "ground_speed");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->ground_speed = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // p_roll_rate
    PyObject * field = PyObject_GetAttrString(_pymsg, "p_roll_rate");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->p_roll_rate = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // q_pitch_rate
    PyObject * field = PyObject_GetAttrString(_pymsg, "q_pitch_rate");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->q_pitch_rate = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // r_yaw_rate
    PyObject * field = PyObject_GetAttrString(_pymsg, "r_yaw_rate");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->r_yaw_rate = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // p_deg_s
    PyObject * field = PyObject_GetAttrString(_pymsg, "p_deg_s");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->p_deg_s = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // q_deg_s
    PyObject * field = PyObject_GetAttrString(_pymsg, "q_deg_s");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->q_deg_s = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // r_deg_s
    PyObject * field = PyObject_GetAttrString(_pymsg, "r_deg_s");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->r_deg_s = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // alpha
    PyObject * field = PyObject_GetAttrString(_pymsg, "alpha");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->alpha = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // beta
    PyObject * field = PyObject_GetAttrString(_pymsg, "beta");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->beta = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // alpha_deg
    PyObject * field = PyObject_GetAttrString(_pymsg, "alpha_deg");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->alpha_deg = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // beta_deg
    PyObject * field = PyObject_GetAttrString(_pymsg, "beta_deg");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->beta_deg = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * my_interfaces__msg__aircraft_data__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of AircraftData */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("my_interfaces.msg._aircraft_data");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "AircraftData");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  my_interfaces__msg__AircraftData * ros_message = (my_interfaces__msg__AircraftData *)raw_ros_message;
  {  // header
    PyObject * field = NULL;
    field = std_msgs__msg__header__convert_to_py(&ros_message->header);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "header", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // north
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->north);
    {
      int rc = PyObject_SetAttrString(_pymessage, "north", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // east
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->east);
    {
      int rc = PyObject_SetAttrString(_pymessage, "east", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // down
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->down);
    {
      int rc = PyObject_SetAttrString(_pymessage, "down", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // altitude
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->altitude);
    {
      int rc = PyObject_SetAttrString(_pymessage, "altitude", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // roll
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->roll);
    {
      int rc = PyObject_SetAttrString(_pymessage, "roll", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // pitch
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->pitch);
    {
      int rc = PyObject_SetAttrString(_pymessage, "pitch", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // yaw
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->yaw);
    {
      int rc = PyObject_SetAttrString(_pymessage, "yaw", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // roll_deg
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->roll_deg);
    {
      int rc = PyObject_SetAttrString(_pymessage, "roll_deg", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // pitch_deg
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->pitch_deg);
    {
      int rc = PyObject_SetAttrString(_pymessage, "pitch_deg", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // yaw_deg
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->yaw_deg);
    {
      int rc = PyObject_SetAttrString(_pymessage, "yaw_deg", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // u_forward
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->u_forward);
    {
      int rc = PyObject_SetAttrString(_pymessage, "u_forward", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // v_sideways
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->v_sideways);
    {
      int rc = PyObject_SetAttrString(_pymessage, "v_sideways", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // w_downward
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->w_downward);
    {
      int rc = PyObject_SetAttrString(_pymessage, "w_downward", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // airspeed
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->airspeed);
    {
      int rc = PyObject_SetAttrString(_pymessage, "airspeed", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // v_north
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->v_north);
    {
      int rc = PyObject_SetAttrString(_pymessage, "v_north", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // v_east
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->v_east);
    {
      int rc = PyObject_SetAttrString(_pymessage, "v_east", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // v_down
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->v_down);
    {
      int rc = PyObject_SetAttrString(_pymessage, "v_down", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // climb_rate
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->climb_rate);
    {
      int rc = PyObject_SetAttrString(_pymessage, "climb_rate", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // ground_speed
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->ground_speed);
    {
      int rc = PyObject_SetAttrString(_pymessage, "ground_speed", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // p_roll_rate
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->p_roll_rate);
    {
      int rc = PyObject_SetAttrString(_pymessage, "p_roll_rate", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // q_pitch_rate
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->q_pitch_rate);
    {
      int rc = PyObject_SetAttrString(_pymessage, "q_pitch_rate", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // r_yaw_rate
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->r_yaw_rate);
    {
      int rc = PyObject_SetAttrString(_pymessage, "r_yaw_rate", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // p_deg_s
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->p_deg_s);
    {
      int rc = PyObject_SetAttrString(_pymessage, "p_deg_s", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // q_deg_s
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->q_deg_s);
    {
      int rc = PyObject_SetAttrString(_pymessage, "q_deg_s", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // r_deg_s
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->r_deg_s);
    {
      int rc = PyObject_SetAttrString(_pymessage, "r_deg_s", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // alpha
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->alpha);
    {
      int rc = PyObject_SetAttrString(_pymessage, "alpha", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // beta
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->beta);
    {
      int rc = PyObject_SetAttrString(_pymessage, "beta", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // alpha_deg
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->alpha_deg);
    {
      int rc = PyObject_SetAttrString(_pymessage, "alpha_deg", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // beta_deg
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->beta_deg);
    {
      int rc = PyObject_SetAttrString(_pymessage, "beta_deg", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
