// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from my_interfaces:msg/CkTable.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__CK_TABLE__BUILDER_HPP_
#define MY_INTERFACES__MSG__DETAIL__CK_TABLE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "my_interfaces/msg/detail/ck_table__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace my_interfaces
{

namespace msg
{

namespace builder
{

class Init_CkTable_position
{
public:
  explicit Init_CkTable_position(::my_interfaces::msg::CkTable & msg)
  : msg_(msg)
  {}
  ::my_interfaces::msg::CkTable position(::my_interfaces::msg::CkTable::_position_type arg)
  {
    msg_.position = std::move(arg);
    return std::move(msg_);
  }

private:
  ::my_interfaces::msg::CkTable msg_;
};

class Init_CkTable_erg_cost_reduction_perc
{
public:
  explicit Init_CkTable_erg_cost_reduction_perc(::my_interfaces::msg::CkTable & msg)
  : msg_(msg)
  {}
  Init_CkTable_position erg_cost_reduction_perc(::my_interfaces::msg::CkTable::_erg_cost_reduction_perc_type arg)
  {
    msg_.erg_cost_reduction_perc = std::move(arg);
    return Init_CkTable_position(msg_);
  }

private:
  ::my_interfaces::msg::CkTable msg_;
};

class Init_CkTable_total_erg_cost_in_range
{
public:
  explicit Init_CkTable_total_erg_cost_in_range(::my_interfaces::msg::CkTable & msg)
  : msg_(msg)
  {}
  Init_CkTable_erg_cost_reduction_perc total_erg_cost_in_range(::my_interfaces::msg::CkTable::_total_erg_cost_in_range_type arg)
  {
    msg_.total_erg_cost_in_range = std::move(arg);
    return Init_CkTable_erg_cost_reduction_perc(msg_);
  }

private:
  ::my_interfaces::msg::CkTable msg_;
};

class Init_CkTable_total_erg_cost
{
public:
  explicit Init_CkTable_total_erg_cost(::my_interfaces::msg::CkTable & msg)
  : msg_(msg)
  {}
  Init_CkTable_total_erg_cost_in_range total_erg_cost(::my_interfaces::msg::CkTable::_total_erg_cost_type arg)
  {
    msg_.total_erg_cost = std::move(arg);
    return Init_CkTable_total_erg_cost_in_range(msg_);
  }

private:
  ::my_interfaces::msg::CkTable msg_;
};

class Init_CkTable_ck_values_average_in_range
{
public:
  explicit Init_CkTable_ck_values_average_in_range(::my_interfaces::msg::CkTable & msg)
  : msg_(msg)
  {}
  Init_CkTable_total_erg_cost ck_values_average_in_range(::my_interfaces::msg::CkTable::_ck_values_average_in_range_type arg)
  {
    msg_.ck_values_average_in_range = std::move(arg);
    return Init_CkTable_total_erg_cost(msg_);
  }

private:
  ::my_interfaces::msg::CkTable msg_;
};

class Init_CkTable_ck_values
{
public:
  explicit Init_CkTable_ck_values(::my_interfaces::msg::CkTable & msg)
  : msg_(msg)
  {}
  Init_CkTable_ck_values_average_in_range ck_values(::my_interfaces::msg::CkTable::_ck_values_type arg)
  {
    msg_.ck_values = std::move(arg);
    return Init_CkTable_ck_values_average_in_range(msg_);
  }

private:
  ::my_interfaces::msg::CkTable msg_;
};

class Init_CkTable_table_size
{
public:
  Init_CkTable_table_size()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CkTable_ck_values table_size(::my_interfaces::msg::CkTable::_table_size_type arg)
  {
    msg_.table_size = std::move(arg);
    return Init_CkTable_ck_values(msg_);
  }

private:
  ::my_interfaces::msg::CkTable msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::my_interfaces::msg::CkTable>()
{
  return my_interfaces::msg::builder::Init_CkTable_table_size();
}

}  // namespace my_interfaces

#endif  // MY_INTERFACES__MSG__DETAIL__CK_TABLE__BUILDER_HPP_
