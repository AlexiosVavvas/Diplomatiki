// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from my_interfaces:msg/MultipleTargetEstimates.idl
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
#include "my_interfaces/msg/detail/multiple_target_estimates__struct.h"
#include "my_interfaces/msg/detail/multiple_target_estimates__functions.h"

#include "rosidl_runtime_c/primitives_sequence.h"
#include "rosidl_runtime_c/primitives_sequence_functions.h"

// Nested array functions includes
#include "my_interfaces/msg/detail/single_target_estimate__functions.h"
// end nested array functions include
ROSIDL_GENERATOR_C_IMPORT
bool std_msgs__msg__header__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * std_msgs__msg__header__convert_to_py(void * raw_ros_message);
bool my_interfaces__msg__single_target_estimate__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * my_interfaces__msg__single_target_estimate__convert_to_py(void * raw_ros_message);
bool my_interfaces__msg__single_target_estimate__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * my_interfaces__msg__single_target_estimate__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool my_interfaces__msg__multiple_target_estimates__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[69];
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
    assert(strncmp("my_interfaces.msg._multiple_target_estimates.MultipleTargetEstimates", full_classname_dest, 68) == 0);
  }
  my_interfaces__msg__MultipleTargetEstimates * ros_message = _ros_message;
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
  {  // num_of_targets
    PyObject * field = PyObject_GetAttrString(_pymsg, "num_of_targets");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->num_of_targets = (int32_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // target_estimates
    PyObject * field = PyObject_GetAttrString(_pymsg, "target_estimates");
    if (!field) {
      return false;
    }
    PyObject * seq_field = PySequence_Fast(field, "expected a sequence in 'target_estimates'");
    if (!seq_field) {
      Py_DECREF(field);
      return false;
    }
    Py_ssize_t size = PySequence_Size(field);
    if (-1 == size) {
      Py_DECREF(seq_field);
      Py_DECREF(field);
      return false;
    }
    if (!my_interfaces__msg__SingleTargetEstimate__Sequence__init(&(ros_message->target_estimates), size)) {
      PyErr_SetString(PyExc_RuntimeError, "unable to create my_interfaces__msg__SingleTargetEstimate__Sequence ros_message");
      Py_DECREF(seq_field);
      Py_DECREF(field);
      return false;
    }
    my_interfaces__msg__SingleTargetEstimate * dest = ros_message->target_estimates.data;
    for (Py_ssize_t i = 0; i < size; ++i) {
      if (!my_interfaces__msg__single_target_estimate__convert_from_py(PySequence_Fast_GET_ITEM(seq_field, i), &dest[i])) {
        Py_DECREF(seq_field);
        Py_DECREF(field);
        return false;
      }
    }
    Py_DECREF(seq_field);
    Py_DECREF(field);
  }
  {  // ground_truths
    PyObject * field = PyObject_GetAttrString(_pymsg, "ground_truths");
    if (!field) {
      return false;
    }
    PyObject * seq_field = PySequence_Fast(field, "expected a sequence in 'ground_truths'");
    if (!seq_field) {
      Py_DECREF(field);
      return false;
    }
    Py_ssize_t size = PySequence_Size(field);
    if (-1 == size) {
      Py_DECREF(seq_field);
      Py_DECREF(field);
      return false;
    }
    if (!my_interfaces__msg__SingleTargetEstimate__Sequence__init(&(ros_message->ground_truths), size)) {
      PyErr_SetString(PyExc_RuntimeError, "unable to create my_interfaces__msg__SingleTargetEstimate__Sequence ros_message");
      Py_DECREF(seq_field);
      Py_DECREF(field);
      return false;
    }
    my_interfaces__msg__SingleTargetEstimate * dest = ros_message->ground_truths.data;
    for (Py_ssize_t i = 0; i < size; ++i) {
      if (!my_interfaces__msg__single_target_estimate__convert_from_py(PySequence_Fast_GET_ITEM(seq_field, i), &dest[i])) {
        Py_DECREF(seq_field);
        Py_DECREF(field);
        return false;
      }
    }
    Py_DECREF(seq_field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * my_interfaces__msg__multiple_target_estimates__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of MultipleTargetEstimates */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("my_interfaces.msg._multiple_target_estimates");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "MultipleTargetEstimates");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  my_interfaces__msg__MultipleTargetEstimates * ros_message = (my_interfaces__msg__MultipleTargetEstimates *)raw_ros_message;
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
  {  // num_of_targets
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->num_of_targets);
    {
      int rc = PyObject_SetAttrString(_pymessage, "num_of_targets", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // target_estimates
    PyObject * field = NULL;
    size_t size = ros_message->target_estimates.size;
    field = PyList_New(size);
    if (!field) {
      return NULL;
    }
    my_interfaces__msg__SingleTargetEstimate * item;
    for (size_t i = 0; i < size; ++i) {
      item = &(ros_message->target_estimates.data[i]);
      PyObject * pyitem = my_interfaces__msg__single_target_estimate__convert_to_py(item);
      if (!pyitem) {
        Py_DECREF(field);
        return NULL;
      }
      int rc = PyList_SetItem(field, i, pyitem);
      (void)rc;
      assert(rc == 0);
    }
    assert(PySequence_Check(field));
    {
      int rc = PyObject_SetAttrString(_pymessage, "target_estimates", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // ground_truths
    PyObject * field = NULL;
    size_t size = ros_message->ground_truths.size;
    field = PyList_New(size);
    if (!field) {
      return NULL;
    }
    my_interfaces__msg__SingleTargetEstimate * item;
    for (size_t i = 0; i < size; ++i) {
      item = &(ros_message->ground_truths.data[i]);
      PyObject * pyitem = my_interfaces__msg__single_target_estimate__convert_to_py(item);
      if (!pyitem) {
        Py_DECREF(field);
        return NULL;
      }
      int rc = PyList_SetItem(field, i, pyitem);
      (void)rc;
      assert(rc == 0);
    }
    assert(PySequence_Check(field));
    {
      int rc = PyObject_SetAttrString(_pymessage, "ground_truths", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
