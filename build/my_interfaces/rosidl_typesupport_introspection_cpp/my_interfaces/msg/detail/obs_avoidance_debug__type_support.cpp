// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from my_interfaces:msg/ObsAvoidanceDebug.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "my_interfaces/msg/detail/obs_avoidance_debug__struct.hpp"
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

void ObsAvoidanceDebug_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) my_interfaces::msg::ObsAvoidanceDebug(_init);
}

void ObsAvoidanceDebug_fini_function(void * message_memory)
{
  auto typed_message = static_cast<my_interfaces::msg::ObsAvoidanceDebug *>(message_memory);
  typed_message->~ObsAvoidanceDebug();
}

size_t size_function__ObsAvoidanceDebug__beta(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<double> *>(untyped_member);
  return member->size();
}

const void * get_const_function__ObsAvoidanceDebug__beta(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<double> *>(untyped_member);
  return &member[index];
}

void * get_function__ObsAvoidanceDebug__beta(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<double> *>(untyped_member);
  return &member[index];
}

void fetch_function__ObsAvoidanceDebug__beta(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__ObsAvoidanceDebug__beta(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__ObsAvoidanceDebug__beta(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__ObsAvoidanceDebug__beta(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

void resize_function__ObsAvoidanceDebug__beta(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<double> *>(untyped_member);
  member->resize(size);
}

size_t size_function__ObsAvoidanceDebug__u_safe(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<double> *>(untyped_member);
  return member->size();
}

const void * get_const_function__ObsAvoidanceDebug__u_safe(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<double> *>(untyped_member);
  return &member[index];
}

void * get_function__ObsAvoidanceDebug__u_safe(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<double> *>(untyped_member);
  return &member[index];
}

void fetch_function__ObsAvoidanceDebug__u_safe(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__ObsAvoidanceDebug__u_safe(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__ObsAvoidanceDebug__u_safe(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__ObsAvoidanceDebug__u_safe(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

void resize_function__ObsAvoidanceDebug__u_safe(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<double> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember ObsAvoidanceDebug_message_member_array[6] = {
  {
    "psi",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces::msg::ObsAvoidanceDebug, psi),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "hddot",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces::msg::ObsAvoidanceDebug, hddot),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "two_alpha_h_hdot",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces::msg::ObsAvoidanceDebug, two_alpha_h_hdot),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "alpha2_h",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces::msg::ObsAvoidanceDebug, alpha2_h),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "beta",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces::msg::ObsAvoidanceDebug, beta),  // bytes offset in struct
    nullptr,  // default value
    size_function__ObsAvoidanceDebug__beta,  // size() function pointer
    get_const_function__ObsAvoidanceDebug__beta,  // get_const(index) function pointer
    get_function__ObsAvoidanceDebug__beta,  // get(index) function pointer
    fetch_function__ObsAvoidanceDebug__beta,  // fetch(index, &value) function pointer
    assign_function__ObsAvoidanceDebug__beta,  // assign(index, value) function pointer
    resize_function__ObsAvoidanceDebug__beta  // resize(index) function pointer
  },
  {
    "u_safe",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(my_interfaces::msg::ObsAvoidanceDebug, u_safe),  // bytes offset in struct
    nullptr,  // default value
    size_function__ObsAvoidanceDebug__u_safe,  // size() function pointer
    get_const_function__ObsAvoidanceDebug__u_safe,  // get_const(index) function pointer
    get_function__ObsAvoidanceDebug__u_safe,  // get(index) function pointer
    fetch_function__ObsAvoidanceDebug__u_safe,  // fetch(index, &value) function pointer
    assign_function__ObsAvoidanceDebug__u_safe,  // assign(index, value) function pointer
    resize_function__ObsAvoidanceDebug__u_safe  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers ObsAvoidanceDebug_message_members = {
  "my_interfaces::msg",  // message namespace
  "ObsAvoidanceDebug",  // message name
  6,  // number of fields
  sizeof(my_interfaces::msg::ObsAvoidanceDebug),
  ObsAvoidanceDebug_message_member_array,  // message members
  ObsAvoidanceDebug_init_function,  // function to initialize message memory (memory has to be allocated)
  ObsAvoidanceDebug_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t ObsAvoidanceDebug_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &ObsAvoidanceDebug_message_members,
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
get_message_type_support_handle<my_interfaces::msg::ObsAvoidanceDebug>()
{
  return &::my_interfaces::msg::rosidl_typesupport_introspection_cpp::ObsAvoidanceDebug_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, my_interfaces, msg, ObsAvoidanceDebug)() {
  return &::my_interfaces::msg::rosidl_typesupport_introspection_cpp::ObsAvoidanceDebug_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
