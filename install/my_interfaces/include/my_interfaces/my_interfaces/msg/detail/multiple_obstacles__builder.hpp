// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from my_interfaces:msg/MultipleObstacles.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__MULTIPLE_OBSTACLES__BUILDER_HPP_
#define MY_INTERFACES__MSG__DETAIL__MULTIPLE_OBSTACLES__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "my_interfaces/msg/detail/multiple_obstacles__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace my_interfaces
{

namespace msg
{

namespace builder
{

class Init_MultipleObstacles_obstacles
{
public:
  explicit Init_MultipleObstacles_obstacles(::my_interfaces::msg::MultipleObstacles & msg)
  : msg_(msg)
  {}
  ::my_interfaces::msg::MultipleObstacles obstacles(::my_interfaces::msg::MultipleObstacles::_obstacles_type arg)
  {
    msg_.obstacles = std::move(arg);
    return std::move(msg_);
  }

private:
  ::my_interfaces::msg::MultipleObstacles msg_;
};

class Init_MultipleObstacles_num_of_obstacles
{
public:
  Init_MultipleObstacles_num_of_obstacles()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MultipleObstacles_obstacles num_of_obstacles(::my_interfaces::msg::MultipleObstacles::_num_of_obstacles_type arg)
  {
    msg_.num_of_obstacles = std::move(arg);
    return Init_MultipleObstacles_obstacles(msg_);
  }

private:
  ::my_interfaces::msg::MultipleObstacles msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::my_interfaces::msg::MultipleObstacles>()
{
  return my_interfaces::msg::builder::Init_MultipleObstacles_num_of_obstacles();
}

}  // namespace my_interfaces

#endif  // MY_INTERFACES__MSG__DETAIL__MULTIPLE_OBSTACLES__BUILDER_HPP_
