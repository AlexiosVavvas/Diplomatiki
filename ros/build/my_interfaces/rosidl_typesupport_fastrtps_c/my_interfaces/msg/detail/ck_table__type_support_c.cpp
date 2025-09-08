// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from my_interfaces:msg/CkTable.idl
// generated code does not contain a copyright notice
#include "my_interfaces/msg/detail/ck_table__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "my_interfaces/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "my_interfaces/msg/detail/ck_table__struct.h"
#include "my_interfaces/msg/detail/ck_table__functions.h"
#include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif

#include "geometry_msgs/msg/detail/point__functions.h"  // position
#include "rosidl_runtime_c/primitives_sequence.h"  // ck_values, ck_values_average_in_range
#include "rosidl_runtime_c/primitives_sequence_functions.h"  // ck_values, ck_values_average_in_range

// forward declare type support functions
ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_my_interfaces
size_t get_serialized_size_geometry_msgs__msg__Point(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_my_interfaces
size_t max_serialized_size_geometry_msgs__msg__Point(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_my_interfaces
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, geometry_msgs, msg, Point)();


using _CkTable__ros_msg_type = my_interfaces__msg__CkTable;

static bool _CkTable__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _CkTable__ros_msg_type * ros_message = static_cast<const _CkTable__ros_msg_type *>(untyped_ros_message);
  // Field name: table_size
  {
    cdr << ros_message->table_size;
  }

  // Field name: ck_values
  {
    size_t size = ros_message->ck_values.size;
    auto array_ptr = ros_message->ck_values.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: ck_values_average_in_range
  {
    size_t size = ros_message->ck_values_average_in_range.size;
    auto array_ptr = ros_message->ck_values_average_in_range.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: total_erg_cost
  {
    cdr << ros_message->total_erg_cost;
  }

  // Field name: total_erg_cost_in_range
  {
    cdr << ros_message->total_erg_cost_in_range;
  }

  // Field name: erg_cost_reduction_perc
  {
    cdr << ros_message->erg_cost_reduction_perc;
  }

  // Field name: position
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, geometry_msgs, msg, Point
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->position, cdr))
    {
      return false;
    }
  }

  return true;
}

static bool _CkTable__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _CkTable__ros_msg_type * ros_message = static_cast<_CkTable__ros_msg_type *>(untyped_ros_message);
  // Field name: table_size
  {
    cdr >> ros_message->table_size;
  }

  // Field name: ck_values
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->ck_values.data) {
      rosidl_runtime_c__double__Sequence__fini(&ros_message->ck_values);
    }
    if (!rosidl_runtime_c__double__Sequence__init(&ros_message->ck_values, size)) {
      fprintf(stderr, "failed to create array for field 'ck_values'");
      return false;
    }
    auto array_ptr = ros_message->ck_values.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: ck_values_average_in_range
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->ck_values_average_in_range.data) {
      rosidl_runtime_c__double__Sequence__fini(&ros_message->ck_values_average_in_range);
    }
    if (!rosidl_runtime_c__double__Sequence__init(&ros_message->ck_values_average_in_range, size)) {
      fprintf(stderr, "failed to create array for field 'ck_values_average_in_range'");
      return false;
    }
    auto array_ptr = ros_message->ck_values_average_in_range.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: total_erg_cost
  {
    cdr >> ros_message->total_erg_cost;
  }

  // Field name: total_erg_cost_in_range
  {
    cdr >> ros_message->total_erg_cost_in_range;
  }

  // Field name: erg_cost_reduction_perc
  {
    cdr >> ros_message->erg_cost_reduction_perc;
  }

  // Field name: position
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, geometry_msgs, msg, Point
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->position))
    {
      return false;
    }
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_my_interfaces
size_t get_serialized_size_my_interfaces__msg__CkTable(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _CkTable__ros_msg_type * ros_message = static_cast<const _CkTable__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name table_size
  {
    size_t item_size = sizeof(ros_message->table_size);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name ck_values
  {
    size_t array_size = ros_message->ck_values.size;
    auto array_ptr = ros_message->ck_values.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name ck_values_average_in_range
  {
    size_t array_size = ros_message->ck_values_average_in_range.size;
    auto array_ptr = ros_message->ck_values_average_in_range.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name total_erg_cost
  {
    size_t item_size = sizeof(ros_message->total_erg_cost);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name total_erg_cost_in_range
  {
    size_t item_size = sizeof(ros_message->total_erg_cost_in_range);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name erg_cost_reduction_perc
  {
    size_t item_size = sizeof(ros_message->erg_cost_reduction_perc);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name position

  current_alignment += get_serialized_size_geometry_msgs__msg__Point(
    &(ros_message->position), current_alignment);

  return current_alignment - initial_alignment;
}

static uint32_t _CkTable__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_my_interfaces__msg__CkTable(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_my_interfaces
size_t max_serialized_size_my_interfaces__msg__CkTable(
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

  // member: table_size
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: ck_values
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
  // member: ck_values_average_in_range
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
  // member: total_erg_cost
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: total_erg_cost_in_range
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: erg_cost_reduction_perc
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: position
  {
    size_t array_size = 1;


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_geometry_msgs__msg__Point(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = my_interfaces__msg__CkTable;
    is_plain =
      (
      offsetof(DataType, position) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _CkTable__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_my_interfaces__msg__CkTable(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_CkTable = {
  "my_interfaces::msg",
  "CkTable",
  _CkTable__cdr_serialize,
  _CkTable__cdr_deserialize,
  _CkTable__get_serialized_size,
  _CkTable__max_serialized_size
};

static rosidl_message_type_support_t _CkTable__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_CkTable,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, my_interfaces, msg, CkTable)() {
  return &_CkTable__type_support;
}

#if defined(__cplusplus)
}
#endif
