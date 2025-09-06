// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from my_interfaces:msg/AgentData.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "my_interfaces/msg/detail/agent_data__rosidl_typesupport_introspection_c.h"
#include "my_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "my_interfaces/msg/detail/agent_data__functions.h"
#include "my_interfaces/msg/detail/agent_data__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `states`
// Member `inputs`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__AgentData_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  my_interfaces__msg__AgentData__init(message_memory);
}

void my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__AgentData_fini_function(void * message_memory)
{
  my_interfaces__msg__AgentData__fini(message_memory);
}

size_t my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__size_function__AgentData__states(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__get_const_function__AgentData__states(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__get_function__AgentData__states(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__fetch_function__AgentData__states(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__get_const_function__AgentData__states(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__assign_function__AgentData__states(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__get_function__AgentData__states(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__resize_function__AgentData__states(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__size_function__AgentData__inputs(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__get_const_function__AgentData__inputs(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__get_function__AgentData__inputs(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__fetch_function__AgentData__inputs(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__get_const_function__AgentData__inputs(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__assign_function__AgentData__inputs(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__get_function__AgentData__inputs(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__resize_function__AgentData__inputs(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__AgentData_message_member_array[8] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__AgentData, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "simulation_time",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__AgentData, simulation_time),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "num_of_states",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__AgentData, num_of_states),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "num_of_inputs",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__AgentData, num_of_inputs),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "states",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__AgentData, states),  // bytes offset in struct
    NULL,  // default value
    my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__size_function__AgentData__states,  // size() function pointer
    my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__get_const_function__AgentData__states,  // get_const(index) function pointer
    my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__get_function__AgentData__states,  // get(index) function pointer
    my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__fetch_function__AgentData__states,  // fetch(index, &value) function pointer
    my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__assign_function__AgentData__states,  // assign(index, value) function pointer
    my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__resize_function__AgentData__states  // resize(index) function pointer
  },
  {
    "inputs",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__AgentData, inputs),  // bytes offset in struct
    NULL,  // default value
    my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__size_function__AgentData__inputs,  // size() function pointer
    my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__get_const_function__AgentData__inputs,  // get_const(index) function pointer
    my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__get_function__AgentData__inputs,  // get(index) function pointer
    my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__fetch_function__AgentData__inputs,  // fetch(index, &value) function pointer
    my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__assign_function__AgentData__inputs,  // assign(index, value) function pointer
    my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__resize_function__AgentData__inputs  // resize(index) function pointer
  },
  {
    "ergodic_cost",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__AgentData, ergodic_cost),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "active_cbf_flag",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__AgentData, active_cbf_flag),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__AgentData_message_members = {
  "my_interfaces__msg",  // message namespace
  "AgentData",  // message name
  8,  // number of fields
  sizeof(my_interfaces__msg__AgentData),
  my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__AgentData_message_member_array,  // message members
  my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__AgentData_init_function,  // function to initialize message memory (memory has to be allocated)
  my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__AgentData_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__AgentData_message_type_support_handle = {
  0,
  &my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__AgentData_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_my_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, my_interfaces, msg, AgentData)() {
  my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__AgentData_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  if (!my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__AgentData_message_type_support_handle.typesupport_identifier) {
    my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__AgentData_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &my_interfaces__msg__AgentData__rosidl_typesupport_introspection_c__AgentData_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
