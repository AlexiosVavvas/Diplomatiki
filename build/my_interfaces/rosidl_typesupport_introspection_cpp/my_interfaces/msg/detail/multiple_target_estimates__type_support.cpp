// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from my_interfaces:msg/MultipleTargetEstimates.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "my_interfaces/msg/detail/multiple_target_estimates__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace my_interfaces
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void MultipleTargetEstimates_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) my_interfaces::msg::MultipleTargetEstimates(_init);
}

void MultipleTargetEstimates_fini_function(void * message_memory)
{
  auto typed_message = static_cast<my_interfaces::msg::MultipleTargetEstimates *>(message_memory);
  typed_message->~MultipleTargetEstimates();
}

size_t size_function__MultipleTargetEstimates__target_estimates(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<my_interfaces::msg::SingleTargetEstimate> *>(untyped_member);
  return member->size();
}

const void * get_const_function__MultipleTargetEstimates__target_estimates(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<my_interfaces::msg::SingleTargetEstimate> *>(untyped_member);
  return &member[index];
}

void * get_function__MultipleTargetEstimates__target_estimates(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<my_interfaces::msg::SingleTargetEstimate> *>(untyped_member);
  return &member[index];
}

void fetch_function__MultipleTargetEstimates__target_estimates(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const my_interfaces::msg::SingleTargetEstimate *>(
    get_const_function__MultipleTargetEstimates__target_estimates(untyped_member, index));
  auto & value = *reinterpret_cast<my_interfaces::msg::SingleTargetEstimate *>(untyped_value);
  value = item;
}

void assign_function__MultipleTargetEstimates__target_estimates(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<my_interfaces::msg::SingleTargetEstimate *>(
    get_function__MultipleTargetEstimates__target_estimates(untyped_member, index));
  const auto & value = *reinterpret_cast<const my_interfaces::msg::SingleTargetEstimate *>(untyped_value);
  item = value;
}

void resize_function__MultipleTargetEstimates__target_estimates(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<my_interfaces::msg::SingleTargetEstimate> *>(untyped_member);
  member->resize(size);
}

size_t size_function__MultipleTargetEstimates__ground_truths(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<my_interfaces::msg::SingleTargetEstimate> *>(untyped_member);
  return member->size();
}

const void * get_const_function__MultipleTargetEstimates__ground_truths(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<my_interfaces::msg::SingleTargetEstimate> *>(untyped_member);
  return &member[index];
}

void * get_function__MultipleTargetEstimates__ground_truths(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<my_interfaces::msg::SingleTargetEstimate> *>(untyped_member);
  return &member[index];
}

void fetch_function__MultipleTargetEstimates__ground_truths(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const my_interfaces::msg::SingleTargetEstimate *>(
    get_const_function__MultipleTargetEstimates__ground_truths(untyped_member, index));
  auto & value = *reinterpret_cast<my_interfaces::msg::SingleTargetEstimate *>(untyped_value);
  value = item;
}

void assign_function__MultipleTargetEstimates__ground_truths(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<my_interfaces::msg::SingleTargetEstimate *>(
    get_function__MultipleTargetEstimates__ground_truths(untyped_member, index));
  const auto & value = *reinterpret_cast<const my_interfaces::msg::SingleTargetEstimate *>(untyped_value);
  item = value;
}

void resize_function__MultipleTargetEstimates__ground_truths(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<my_interfaces::msg::SingleTargetEstimate> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember MultipleTargetEstimates_message_member_array[4] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces::msg::MultipleTargetEstimates, header),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "num_of_targets",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces::msg::MultipleTargetEstimates, num_of_targets),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "target_estimates",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<my_interfaces::msg::SingleTargetEstimate>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces::msg::MultipleTargetEstimates, target_estimates),  // bytes offset in struct
    nullptr,  // default value
    size_function__MultipleTargetEstimates__target_estimates,  // size() function pointer
    get_const_function__MultipleTargetEstimates__target_estimates,  // get_const(index) function pointer
    get_function__MultipleTargetEstimates__target_estimates,  // get(index) function pointer
    fetch_function__MultipleTargetEstimates__target_estimates,  // fetch(index, &value) function pointer
    assign_function__MultipleTargetEstimates__target_estimates,  // assign(index, value) function pointer
    resize_function__MultipleTargetEstimates__target_estimates  // resize(index) function pointer
  },
  {
    "ground_truths",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<my_interfaces::msg::SingleTargetEstimate>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces::msg::MultipleTargetEstimates, ground_truths),  // bytes offset in struct
    nullptr,  // default value
    size_function__MultipleTargetEstimates__ground_truths,  // size() function pointer
    get_const_function__MultipleTargetEstimates__ground_truths,  // get_const(index) function pointer
    get_function__MultipleTargetEstimates__ground_truths,  // get(index) function pointer
    fetch_function__MultipleTargetEstimates__ground_truths,  // fetch(index, &value) function pointer
    assign_function__MultipleTargetEstimates__ground_truths,  // assign(index, value) function pointer
    resize_function__MultipleTargetEstimates__ground_truths  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers MultipleTargetEstimates_message_members = {
  "my_interfaces::msg",  // message namespace
  "MultipleTargetEstimates",  // message name
  4,  // number of fields
  sizeof(my_interfaces::msg::MultipleTargetEstimates),
  MultipleTargetEstimates_message_member_array,  // message members
  MultipleTargetEstimates_init_function,  // function to initialize message memory (memory has to be allocated)
  MultipleTargetEstimates_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t MultipleTargetEstimates_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &MultipleTargetEstimates_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace my_interfaces


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<my_interfaces::msg::MultipleTargetEstimates>()
{
  return &::my_interfaces::msg::rosidl_typesupport_introspection_cpp::MultipleTargetEstimates_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, my_interfaces, msg, MultipleTargetEstimates)() {
  return &::my_interfaces::msg::rosidl_typesupport_introspection_cpp::MultipleTargetEstimates_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
