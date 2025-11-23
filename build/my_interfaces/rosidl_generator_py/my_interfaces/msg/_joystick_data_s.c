// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from my_interfaces:msg/JoystickData.idl
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
#include "my_interfaces/msg/detail/joystick_data__struct.h"
#include "my_interfaces/msg/detail/joystick_data__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool my_interfaces__msg__joystick_data__convert_from_py(PyObject * _pymsg, void * _ros_message)
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
    assert(strncmp("my_interfaces.msg._joystick_data.JoystickData", full_classname_dest, 45) == 0);
  }
  my_interfaces__msg__JoystickData * ros_message = _ros_message;
  {  // throttle
    PyObject * field = PyObject_GetAttrString(_pymsg, "throttle");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->throttle = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // aileron
    PyObject * field = PyObject_GetAttrString(_pymsg, "aileron");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->aileron = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // elevator
    PyObject * field = PyObject_GetAttrString(_pymsg, "elevator");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->elevator = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // rudder
    PyObject * field = PyObject_GetAttrString(_pymsg, "rudder");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->rudder = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // switch_state
    PyObject * field = PyObject_GetAttrString(_pymsg, "switch_state");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->switch_state = (int32_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * my_interfaces__msg__joystick_data__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of JoystickData */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("my_interfaces.msg._joystick_data");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "JoystickData");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  my_interfaces__msg__JoystickData * ros_message = (my_interfaces__msg__JoystickData *)raw_ros_message;
  {  // throttle
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->throttle);
    {
      int rc = PyObject_SetAttrString(_pymessage, "throttle", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // aileron
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->aileron);
    {
      int rc = PyObject_SetAttrString(_pymessage, "aileron", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // elevator
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->elevator);
    {
      int rc = PyObject_SetAttrString(_pymessage, "elevator", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // rudder
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->rudder);
    {
      int rc = PyObject_SetAttrString(_pymessage, "rudder", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // switch_state
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->switch_state);
    {
      int rc = PyObject_SetAttrString(_pymessage, "switch_state", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
