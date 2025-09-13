// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from my_interfaces:msg/MultipleTargetEstimates.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "my_interfaces/msg/detail/multiple_target_estimates__rosidl_typesupport_introspection_c.h"
#include "my_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "my_interfaces/msg/detail/multiple_target_estimates__functions.h"
#include "my_interfaces/msg/detail/multiple_target_estimates__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `target_estimates`
// Member `ground_truths`
#include "my_interfaces/msg/single_target_estimate.h"
// Member `target_estimates`
// Member `ground_truths`
#include "my_interfaces/msg/detail/single_target_estimate__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__MultipleTargetEstimates_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  my_interfaces__msg__MultipleTargetEstimates__init(message_memory);
}

void my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__MultipleTargetEstimates_fini_function(void * message_memory)
{
  my_interfaces__msg__MultipleTargetEstimates__fini(message_memory);
}

size_t my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__size_function__MultipleTargetEstimates__target_estimates(
  const void * untyped_member)
{
  const my_interfaces__msg__SingleTargetEstimate__Sequence * member =
    (const my_interfaces__msg__SingleTargetEstimate__Sequence *)(untyped_member);
  return member->size;
}

const void * my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__get_const_function__MultipleTargetEstimates__target_estimates(
  const void * untyped_member, size_t index)
{
  const my_interfaces__msg__SingleTargetEstimate__Sequence * member =
    (const my_interfaces__msg__SingleTargetEstimate__Sequence *)(untyped_member);
  return &member->data[index];
}

void * my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__get_function__MultipleTargetEstimates__target_estimates(
  void * untyped_member, size_t index)
{
  my_interfaces__msg__SingleTargetEstimate__Sequence * member =
    (my_interfaces__msg__SingleTargetEstimate__Sequence *)(untyped_member);
  return &member->data[index];
}

void my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__fetch_function__MultipleTargetEstimates__target_estimates(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const my_interfaces__msg__SingleTargetEstimate * item =
    ((const my_interfaces__msg__SingleTargetEstimate *)
    my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__get_const_function__MultipleTargetEstimates__target_estimates(untyped_member, index));
  my_interfaces__msg__SingleTargetEstimate * value =
    (my_interfaces__msg__SingleTargetEstimate *)(untyped_value);
  *value = *item;
}

void my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__assign_function__MultipleTargetEstimates__target_estimates(
  void * untyped_member, size_t index, const void * untyped_value)
{
  my_interfaces__msg__SingleTargetEstimate * item =
    ((my_interfaces__msg__SingleTargetEstimate *)
    my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__get_function__MultipleTargetEstimates__target_estimates(untyped_member, index));
  const my_interfaces__msg__SingleTargetEstimate * value =
    (const my_interfaces__msg__SingleTargetEstimate *)(untyped_value);
  *item = *value;
}

bool my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__resize_function__MultipleTargetEstimates__target_estimates(
  void * untyped_member, size_t size)
{
  my_interfaces__msg__SingleTargetEstimate__Sequence * member =
    (my_interfaces__msg__SingleTargetEstimate__Sequence *)(untyped_member);
  my_interfaces__msg__SingleTargetEstimate__Sequence__fini(member);
  return my_interfaces__msg__SingleTargetEstimate__Sequence__init(member, size);
}

size_t my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__size_function__MultipleTargetEstimates__ground_truths(
  const void * untyped_member)
{
  const my_interfaces__msg__SingleTargetEstimate__Sequence * member =
    (const my_interfaces__msg__SingleTargetEstimate__Sequence *)(untyped_member);
  return member->size;
}

const void * my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__get_const_function__MultipleTargetEstimates__ground_truths(
  const void * untyped_member, size_t index)
{
  const my_interfaces__msg__SingleTargetEstimate__Sequence * member =
    (const my_interfaces__msg__SingleTargetEstimate__Sequence *)(untyped_member);
  return &member->data[index];
}

void * my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__get_function__MultipleTargetEstimates__ground_truths(
  void * untyped_member, size_t index)
{
  my_interfaces__msg__SingleTargetEstimate__Sequence * member =
    (my_interfaces__msg__SingleTargetEstimate__Sequence *)(untyped_member);
  return &member->data[index];
}

void my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__fetch_function__MultipleTargetEstimates__ground_truths(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const my_interfaces__msg__SingleTargetEstimate * item =
    ((const my_interfaces__msg__SingleTargetEstimate *)
    my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__get_const_function__MultipleTargetEstimates__ground_truths(untyped_member, index));
  my_interfaces__msg__SingleTargetEstimate * value =
    (my_interfaces__msg__SingleTargetEstimate *)(untyped_value);
  *value = *item;
}

void my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__assign_function__MultipleTargetEstimates__ground_truths(
  void * untyped_member, size_t index, const void * untyped_value)
{
  my_interfaces__msg__SingleTargetEstimate * item =
    ((my_interfaces__msg__SingleTargetEstimate *)
    my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__get_function__MultipleTargetEstimates__ground_truths(untyped_member, index));
  const my_interfaces__msg__SingleTargetEstimate * value =
    (const my_interfaces__msg__SingleTargetEstimate *)(untyped_value);
  *item = *value;
}

bool my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__resize_function__MultipleTargetEstimates__ground_truths(
  void * untyped_member, size_t size)
{
  my_interfaces__msg__SingleTargetEstimate__Sequence * member =
    (my_interfaces__msg__SingleTargetEstimate__Sequence *)(untyped_member);
  my_interfaces__msg__SingleTargetEstimate__Sequence__fini(member);
  return my_interfaces__msg__SingleTargetEstimate__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__MultipleTargetEstimates_message_member_array[4] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__MultipleTargetEstimates, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "num_of_targets",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__MultipleTargetEstimates, num_of_targets),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "target_estimates",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__MultipleTargetEstimates, target_estimates),  // bytes offset in struct
    NULL,  // default value
    my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__size_function__MultipleTargetEstimates__target_estimates,  // size() function pointer
    my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__get_const_function__MultipleTargetEstimates__target_estimates,  // get_const(index) function pointer
    my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__get_function__MultipleTargetEstimates__target_estimates,  // get(index) function pointer
    my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__fetch_function__MultipleTargetEstimates__target_estimates,  // fetch(index, &value) function pointer
    my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__assign_function__MultipleTargetEstimates__target_estimates,  // assign(index, value) function pointer
    my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__resize_function__MultipleTargetEstimates__target_estimates  // resize(index) function pointer
  },
  {
    "ground_truths",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces__msg__MultipleTargetEstimates, ground_truths),  // bytes offset in struct
    NULL,  // default value
    my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__size_function__MultipleTargetEstimates__ground_truths,  // size() function pointer
    my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__get_const_function__MultipleTargetEstimates__ground_truths,  // get_const(index) function pointer
    my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__get_function__MultipleTargetEstimates__ground_truths,  // get(index) function pointer
    my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__fetch_function__MultipleTargetEstimates__ground_truths,  // fetch(index, &value) function pointer
    my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__assign_function__MultipleTargetEstimates__ground_truths,  // assign(index, value) function pointer
    my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__resize_function__MultipleTargetEstimates__ground_truths  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__MultipleTargetEstimates_message_members = {
  "my_interfaces__msg",  // message namespace
  "MultipleTargetEstimates",  // message name
  4,  // number of fields
  sizeof(my_interfaces__msg__MultipleTargetEstimates),
  my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__MultipleTargetEstimates_message_member_array,  // message members
  my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__MultipleTargetEstimates_init_function,  // function to initialize message memory (memory has to be allocated)
  my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__MultipleTargetEstimates_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__MultipleTargetEstimates_message_type_support_handle = {
  0,
  &my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__MultipleTargetEstimates_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_my_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, my_interfaces, msg, MultipleTargetEstimates)() {
  my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__MultipleTargetEstimates_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__MultipleTargetEstimates_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, my_interfaces, msg, SingleTargetEstimate)();
  my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__MultipleTargetEstimates_message_member_array[3].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, my_interfaces, msg, SingleTargetEstimate)();
  if (!my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__MultipleTargetEstimates_message_type_support_handle.typesupport_identifier) {
    my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__MultipleTargetEstimates_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &my_interfaces__msg__MultipleTargetEstimates__rosidl_typesupport_introspection_c__MultipleTargetEstimates_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
