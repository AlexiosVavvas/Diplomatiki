// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from my_interfaces:msg/CkTable.idl
// generated code does not contain a copyright notice

#ifndef MY_INTERFACES__MSG__DETAIL__CK_TABLE__STRUCT_HPP_
#define MY_INTERFACES__MSG__DETAIL__CK_TABLE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__my_interfaces__msg__CkTable __attribute__((deprecated))
#else
# define DEPRECATED__my_interfaces__msg__CkTable __declspec(deprecated)
#endif

namespace my_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct CkTable_
{
  using Type = CkTable_<ContainerAllocator>;

  explicit CkTable_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->table_size = 0l;
      this->total_erg_cost = 0.0;
    }
  }

  explicit CkTable_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->table_size = 0l;
      this->total_erg_cost = 0.0;
    }
  }

  // field types and members
  using _table_size_type =
    int32_t;
  _table_size_type table_size;
  using _ck_values_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _ck_values_type ck_values;
  using _total_erg_cost_type =
    double;
  _total_erg_cost_type total_erg_cost;

  // setters for named parameter idiom
  Type & set__table_size(
    const int32_t & _arg)
  {
    this->table_size = _arg;
    return *this;
  }
  Type & set__ck_values(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->ck_values = _arg;
    return *this;
  }
  Type & set__total_erg_cost(
    const double & _arg)
  {
    this->total_erg_cost = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    my_interfaces::msg::CkTable_<ContainerAllocator> *;
  using ConstRawPtr =
    const my_interfaces::msg::CkTable_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<my_interfaces::msg::CkTable_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<my_interfaces::msg::CkTable_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      my_interfaces::msg::CkTable_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<my_interfaces::msg::CkTable_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      my_interfaces::msg::CkTable_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<my_interfaces::msg::CkTable_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<my_interfaces::msg::CkTable_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<my_interfaces::msg::CkTable_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__my_interfaces__msg__CkTable
    std::shared_ptr<my_interfaces::msg::CkTable_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__my_interfaces__msg__CkTable
    std::shared_ptr<my_interfaces::msg::CkTable_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const CkTable_ & other) const
  {
    if (this->table_size != other.table_size) {
      return false;
    }
    if (this->ck_values != other.ck_values) {
      return false;
    }
    if (this->total_erg_cost != other.total_erg_cost) {
      return false;
    }
    return true;
  }
  bool operator!=(const CkTable_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct CkTable_

// alias to use template instance with default allocator
using CkTable =
  my_interfaces::msg::CkTable_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace my_interfaces

#endif  // MY_INTERFACES__MSG__DETAIL__CK_TABLE__STRUCT_HPP_
