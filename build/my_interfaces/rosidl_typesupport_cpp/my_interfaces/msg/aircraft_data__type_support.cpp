// generated from rosidl_typesupport_cpp/resource/idl__type_support.cpp.em
// with input from my_interfaces:msg/AircraftData.idl
// generated code does not contain a copyright notice

#include "cstddef"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "my_interfaces/msg/detail/aircraft_data__struct.hpp"
#include "rosidl_typesupport_cpp/identifier.hpp"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_c/type_support_map.h"
#include "rosidl_typesupport_cpp/message_type_support_dispatch.hpp"
#include "rosidl_typesupport_cpp/visibility_control.h"
#include "rosidl_typesupport_interface/macros.h"

namespace my_interfaces
{

namespace msg
{

namespace rosidl_typesupport_cpp
{

typedef struct _AircraftData_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _AircraftData_type_support_ids_t;

static const _AircraftData_type_support_ids_t _AircraftData_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  }
};

typedef struct _AircraftData_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _AircraftData_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _AircraftData_type_support_symbol_names_t _AircraftData_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, my_interfaces, msg, AircraftData)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, my_interfaces, msg, AircraftData)),
  }
};

typedef struct _AircraftData_type_support_data_t
{
  void * data[2];
} _AircraftData_type_support_data_t;

static _AircraftData_type_support_data_t _AircraftData_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _AircraftData_message_typesupport_map = {
  2,
  "my_interfaces",
  &_AircraftData_message_typesupport_ids.typesupport_identifier[0],
  &_AircraftData_message_typesupport_symbol_names.symbol_name[0],
  &_AircraftData_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t AircraftData_message_type_support_handle = {
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_AircraftData_message_typesupport_map),
  ::rosidl_typesupport_cpp::get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace msg

}  // namespace my_interfaces

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<my_interfaces::msg::AircraftData>()
{
  return &::my_interfaces::msg::rosidl_typesupport_cpp::AircraftData_message_type_support_handle;
}

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_cpp, my_interfaces, msg, AircraftData)() {
  return get_message_type_support_handle<my_interfaces::msg::AircraftData>();
}

#ifdef __cplusplus
}
#endif
}  // namespace rosidl_typesupport_cpp
