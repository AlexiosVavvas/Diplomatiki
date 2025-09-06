// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from my_interfaces:msg/MultipleTargetEstimates.idl
// generated code does not contain a copyright notice
#include "my_interfaces/msg/detail/multiple_target_estimates__rosidl_typesupport_fastrtps_cpp.hpp"
#include "my_interfaces/msg/detail/multiple_target_estimates__struct.hpp"

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
namespace std_msgs
{
namespace msg
{
namespace typesupport_fastrtps_cpp
{
bool cdr_serialize(
  const std_msgs::msg::Header &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  std_msgs::msg::Header &);
size_t get_serialized_size(
  const std_msgs::msg::Header &,
  size_t current_alignment);
size_t
max_serialized_size_Header(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);
}  // namespace typesupport_fastrtps_cpp
}  // namespace msg
}  // namespace std_msgs

namespace my_interfaces
{
namespace msg
{
namespace typesupport_fastrtps_cpp
{
bool cdr_serialize(
  const my_interfaces::msg::SingleTargetEstimate &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  my_interfaces::msg::SingleTargetEstimate &);
size_t get_serialized_size(
  const my_interfaces::msg::SingleTargetEstimate &,
  size_t current_alignment);
size_t
max_serialized_size_SingleTargetEstimate(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);
}  // namespace typesupport_fastrtps_cpp
}  // namespace msg
}  // namespace my_interfaces

// functions for my_interfaces::msg::SingleTargetEstimate already declared above


namespace my_interfaces
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_my_interfaces
cdr_serialize(
  const my_interfaces::msg::MultipleTargetEstimates & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: header
  std_msgs::msg::typesupport_fastrtps_cpp::cdr_serialize(
    ros_message.header,
    cdr);
  // Member: num_of_targets
  cdr << ros_message.num_of_targets;
  // Member: target_estimates
  {
    size_t size = ros_message.target_estimates.size();
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; i++) {
      my_interfaces::msg::typesupport_fastrtps_cpp::cdr_serialize(
        ros_message.target_estimates[i],
        cdr);
    }
  }
  // Member: ground_truths
  {
    size_t size = ros_message.ground_truths.size();
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; i++) {
      my_interfaces::msg::typesupport_fastrtps_cpp::cdr_serialize(
        ros_message.ground_truths[i],
        cdr);
    }
  }
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_my_interfaces
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  my_interfaces::msg::MultipleTargetEstimates & ros_message)
{
  // Member: header
  std_msgs::msg::typesupport_fastrtps_cpp::cdr_deserialize(
    cdr, ros_message.header);

  // Member: num_of_targets
  cdr >> ros_message.num_of_targets;

  // Member: target_estimates
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    ros_message.target_estimates.resize(size);
    for (size_t i = 0; i < size; i++) {
      my_interfaces::msg::typesupport_fastrtps_cpp::cdr_deserialize(
        cdr, ros_message.target_estimates[i]);
    }
  }

  // Member: ground_truths
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    ros_message.ground_truths.resize(size);
    for (size_t i = 0; i < size; i++) {
      my_interfaces::msg::typesupport_fastrtps_cpp::cdr_deserialize(
        cdr, ros_message.ground_truths[i]);
    }
  }

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_my_interfaces
get_serialized_size(
  const my_interfaces::msg::MultipleTargetEstimates & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: header

  current_alignment +=
    std_msgs::msg::typesupport_fastrtps_cpp::get_serialized_size(
    ros_message.header, current_alignment);
  // Member: num_of_targets
  {
    size_t item_size = sizeof(ros_message.num_of_targets);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: target_estimates
  {
    size_t array_size = ros_message.target_estimates.size();

    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        my_interfaces::msg::typesupport_fastrtps_cpp::get_serialized_size(
        ros_message.target_estimates[index], current_alignment);
    }
  }
  // Member: ground_truths
  {
    size_t array_size = ros_message.ground_truths.size();

    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        my_interfaces::msg::typesupport_fastrtps_cpp::get_serialized_size(
        ros_message.ground_truths[index], current_alignment);
    }
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_my_interfaces
max_serialized_size_MultipleTargetEstimates(
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


  // Member: header
  {
    size_t array_size = 1;


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size =
        std_msgs::msg::typesupport_fastrtps_cpp::max_serialized_size_Header(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Member: num_of_targets
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: target_estimates
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size =
        my_interfaces::msg::typesupport_fastrtps_cpp::max_serialized_size_SingleTargetEstimate(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Member: ground_truths
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size =
        my_interfaces::msg::typesupport_fastrtps_cpp::max_serialized_size_SingleTargetEstimate(
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
    using DataType = my_interfaces::msg::MultipleTargetEstimates;
    is_plain =
      (
      offsetof(DataType, ground_truths) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static bool _MultipleTargetEstimates__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const my_interfaces::msg::MultipleTargetEstimates *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _MultipleTargetEstimates__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<my_interfaces::msg::MultipleTargetEstimates *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _MultipleTargetEstimates__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const my_interfaces::msg::MultipleTargetEstimates *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _MultipleTargetEstimates__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_MultipleTargetEstimates(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _MultipleTargetEstimates__callbacks = {
  "my_interfaces::msg",
  "MultipleTargetEstimates",
  _MultipleTargetEstimates__cdr_serialize,
  _MultipleTargetEstimates__cdr_deserialize,
  _MultipleTargetEstimates__get_serialized_size,
  _MultipleTargetEstimates__max_serialized_size
};

static rosidl_message_type_support_t _MultipleTargetEstimates__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_MultipleTargetEstimates__callbacks,
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
get_message_type_support_handle<my_interfaces::msg::MultipleTargetEstimates>()
{
  return &my_interfaces::msg::typesupport_fastrtps_cpp::_MultipleTargetEstimates__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, my_interfaces, msg, MultipleTargetEstimates)() {
  return &my_interfaces::msg::typesupport_fastrtps_cpp::_MultipleTargetEstimates__handle;
}

#ifdef __cplusplus
}
#endif
