// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from my_interfaces:msg/CkTable.idl
// generated code does not contain a copyright notice
#include "my_interfaces/msg/detail/ck_table__rosidl_typesupport_fastrtps_cpp.hpp"
#include "my_interfaces/msg/detail/ck_table__struct.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_fastrtps_cpp/identifier.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_fastrtps_cpp/wstring_conversion.hpp"
#include "fastcdr/Cdr.h"


// forward declaration of message dependencies and their conversion functions

namespace my_interfaces
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_my_interfaces
cdr_serialize(
  const my_interfaces::msg::CkTable & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: table_size
  cdr << ros_message.table_size;
  // Member: ck_values
  {
    cdr << ros_message.ck_values;
  }
  // Member: total_erg_cost
  cdr << ros_message.total_erg_cost;
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_my_interfaces
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  my_interfaces::msg::CkTable & ros_message)
{
  // Member: table_size
  cdr >> ros_message.table_size;

  // Member: ck_values
  {
    cdr >> ros_message.ck_values;
  }

  // Member: total_erg_cost
  cdr >> ros_message.total_erg_cost;

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_my_interfaces
get_serialized_size(
  const my_interfaces::msg::CkTable & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: table_size
  {
    size_t item_size = sizeof(ros_message.table_size);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: ck_values
  {
    size_t array_size = ros_message.ck_values.size();

    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    size_t item_size = sizeof(ros_message.ck_values[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: total_erg_cost
  {
    size_t item_size = sizeof(ros_message.total_erg_cost);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_my_interfaces
max_serialized_size_CkTable(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;


  // Member: table_size
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: ck_values
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: total_erg_cost
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = my_interfaces::msg::CkTable;
    is_plain =
      (
      offsetof(DataType, total_erg_cost) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static bool _CkTable__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const my_interfaces::msg::CkTable *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _CkTable__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<my_interfaces::msg::CkTable *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _CkTable__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const my_interfaces::msg::CkTable *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _CkTable__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_CkTable(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _CkTable__callbacks = {
  "my_interfaces::msg",
  "CkTable",
  _CkTable__cdr_serialize,
  _CkTable__cdr_deserialize,
  _CkTable__get_serialized_size,
  _CkTable__max_serialized_size
};

static rosidl_message_type_support_t _CkTable__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_CkTable__callbacks,
  get_message_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace my_interfaces

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_my_interfaces
const rosidl_message_type_support_t *
get_message_type_support_handle<my_interfaces::msg::CkTable>()
{
  return &my_interfaces::msg::typesupport_fastrtps_cpp::_CkTable__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, my_interfaces, msg, CkTable)() {
  return &my_interfaces::msg::typesupport_fastrtps_cpp::_CkTable__handle;
}

#ifdef __cplusplus
}
#endif
