// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from my_interfaces:msg/CkTable.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "my_interfaces/msg/detail/ck_table__rosidl_typesupport_introspection_c.h"
#include "my_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "my_interfaces/msg/detail/ck_table__functions.h"
#include "my_interfaces/msg/detail/ck_table__struct.h"


// Include directives for member types
// Member `model_type`
#include "rosidl_runtime_c/string_functions.h"
// Member `l_bounds`
// Member `ck_values`
// Member `ck_values_average_in_range`
#include "rosidl_runtime_c/primitives_sequence_functions.h"
// Member `position`
#include "geometry_msgs/msg/point.h"
// Member `position`
#include "geometry_msgs/msg/detail/point__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__CkTable_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  my_interfaces__msg__CkTable__init(message_memory);
}

void my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__CkTable_fini_function(void * message_memory)
{
  my_interfaces__msg__CkTable__fini(message_memory);
}

size_t my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__size_function__CkTable__l_bounds(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__get_const_function__CkTable__l_bounds(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__get_function__CkTable__l_bounds(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__fetch_function__CkTable__l_bounds(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__get_const_function__CkTable__l_bounds(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__assign_function__CkTable__l_bounds(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__get_function__CkTable__l_bounds(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__resize_function__CkTable__l_bounds(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__size_function__CkTable__ck_values(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__get_const_function__CkTable__ck_values(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__get_function__CkTable__ck_values(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__fetch_function__CkTable__ck_values(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__get_const_function__CkTable__ck_values(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__assign_function__CkTable__ck_values(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__get_function__CkTable__ck_values(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__resize_function__CkTable__ck_values(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__size_function__CkTable__ck_values_average_in_range(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__get_const_function__CkTable__ck_values_average_in_range(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__get_function__CkTable__ck_values_average_in_range(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__fetch_function__CkTable__ck_values_average_in_range(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__get_const_function__CkTable__ck_values_average_in_range(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__assign_function__CkTable__ck_values_average_in_range(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__get_function__CkTable__ck_values_average_in_range(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__resize_function__CkTable__ck_values_average_in_range(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__CkTable_message_member_array[9] = {
  {
    "model_type",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__CkTable, model_type),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "l_bounds",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__CkTable, l_bounds),  // bytes offset in struct
    NULL,  // default value
    my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__size_function__CkTable__l_bounds,  // size() function pointer
    my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__get_const_function__CkTable__l_bounds,  // get_const(index) function pointer
    my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__get_function__CkTable__l_bounds,  // get(index) function pointer
    my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__fetch_function__CkTable__l_bounds,  // fetch(index, &value) function pointer
    my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__assign_function__CkTable__l_bounds,  // assign(index, value) function pointer
    my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__resize_function__CkTable__l_bounds  // resize(index) function pointer
  },
  {
    "table_size",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__CkTable, table_size),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "ck_values",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__CkTable, ck_values),  // bytes offset in struct
    NULL,  // default value
    my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__size_function__CkTable__ck_values,  // size() function pointer
    my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__get_const_function__CkTable__ck_values,  // get_const(index) function pointer
    my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__get_function__CkTable__ck_values,  // get(index) function pointer
    my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__fetch_function__CkTable__ck_values,  // fetch(index, &value) function pointer
    my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__assign_function__CkTable__ck_values,  // assign(index, value) function pointer
    my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__resize_function__CkTable__ck_values  // resize(index) function pointer
  },
  {
    "ck_values_average_in_range",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__CkTable, ck_values_average_in_range),  // bytes offset in struct
    NULL,  // default value
    my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__size_function__CkTable__ck_values_average_in_range,  // size() function pointer
    my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__get_const_function__CkTable__ck_values_average_in_range,  // get_const(index) function pointer
    my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__get_function__CkTable__ck_values_average_in_range,  // get(index) function pointer
    my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__fetch_function__CkTable__ck_values_average_in_range,  // fetch(index, &value) function pointer
    my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__assign_function__CkTable__ck_values_average_in_range,  // assign(index, value) function pointer
    my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__resize_function__CkTable__ck_values_average_in_range  // resize(index) function pointer
  },
  {
    "total_erg_cost",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__CkTable, total_erg_cost),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "total_erg_cost_in_range",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__CkTable, total_erg_cost_in_range),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "erg_cost_reduction_perc",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__CkTable, erg_cost_reduction_perc),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "position",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__CkTable, position),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__CkTable_message_members = {
  "my_interfaces__msg",  // message namespace
  "CkTable",  // message name
  9,  // number of fields
  sizeof(my_interfaces__msg__CkTable),
  my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__CkTable_message_member_array,  // message members
  my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__CkTable_init_function,  // function to initialize message memory (memory has to be allocated)
  my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__CkTable_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__CkTable_message_type_support_handle = {
  0,
  &my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__CkTable_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_my_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, my_interfaces, msg, CkTable)() {
  my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__CkTable_message_member_array[8].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Point)();
  if (!my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__CkTable_message_type_support_handle.typesupport_identifier) {
    my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__CkTable_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &my_interfaces__msg__CkTable__rosidl_typesupport_introspection_c__CkTable_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
