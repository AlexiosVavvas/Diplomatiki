// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from my_interfaces:msg/SingleObstacle.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "my_interfaces/msg/detail/single_obstacle__rosidl_typesupport_introspection_c.h"
#include "my_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "my_interfaces/msg/detail/single_obstacle__functions.h"
#include "my_interfaces/msg/detail/single_obstacle__struct.h"


// Include directives for member types
// Member `obs_type`
// Member `obs_name`
#include "rosidl_runtime_c/string_functions.h"
// Member `position`
#include "geometry_msgs/msg/point.h"
// Member `position`
#include "geometry_msgs/msg/detail/point__rosidl_typesupport_introspection_c.h"
// Member `dimensions`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__SingleObstacle_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  my_interfaces__msg__SingleObstacle__init(message_memory);
}

void my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__SingleObstacle_fini_function(void * message_memory)
{
  my_interfaces__msg__SingleObstacle__fini(message_memory);
}

size_t my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__size_function__SingleObstacle__dimensions(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__get_const_function__SingleObstacle__dimensions(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__get_function__SingleObstacle__dimensions(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__fetch_function__SingleObstacle__dimensions(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__get_const_function__SingleObstacle__dimensions(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__assign_function__SingleObstacle__dimensions(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__get_function__SingleObstacle__dimensions(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__resize_function__SingleObstacle__dimensions(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__SingleObstacle_message_member_array[6] = {
  {
    "obs_type",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__SingleObstacle, obs_type),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "obs_name",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__SingleObstacle, obs_name),  // bytes offset in struct
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
    offsetof(my_interfaces__msg__SingleObstacle, position),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "dimensions",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__SingleObstacle, dimensions),  // bytes offset in struct
    NULL,  // default value
    my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__size_function__SingleObstacle__dimensions,  // size() function pointer
    my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__get_const_function__SingleObstacle__dimensions,  // get_const(index) function pointer
    my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__get_function__SingleObstacle__dimensions,  // get(index) function pointer
    my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__fetch_function__SingleObstacle__dimensions,  // fetch(index, &value) function pointer
    my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__assign_function__SingleObstacle__dimensions,  // assign(index, value) function pointer
    my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__resize_function__SingleObstacle__dimensions  // resize(index) function pointer
  },
  {
    "kappa",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__SingleObstacle, kappa),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "rho0",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__SingleObstacle, rho0),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__SingleObstacle_message_members = {
  "my_interfaces__msg",  // message namespace
  "SingleObstacle",  // message name
  6,  // number of fields
  sizeof(my_interfaces__msg__SingleObstacle),
  my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__SingleObstacle_message_member_array,  // message members
  my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__SingleObstacle_init_function,  // function to initialize message memory (memory has to be allocated)
  my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__SingleObstacle_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__SingleObstacle_message_type_support_handle = {
  0,
  &my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__SingleObstacle_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_my_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, my_interfaces, msg, SingleObstacle)() {
  my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__SingleObstacle_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Point)();
  if (!my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__SingleObstacle_message_type_support_handle.typesupport_identifier) {
    my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__SingleObstacle_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &my_interfaces__msg__SingleObstacle__rosidl_typesupport_introspection_c__SingleObstacle_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
