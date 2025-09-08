// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from my_interfaces:msg/CkTable.idl
// generated code does not contain a copyright notice
#include "my_interfaces/msg/detail/ck_table__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `ck_values`
// Member `ck_values_average_in_range`
#include "rosidl_runtime_c/primitives_sequence_functions.h"
// Member `position`
#include "geometry_msgs/msg/detail/point__functions.h"

bool
my_interfaces__msg__CkTable__init(my_interfaces__msg__CkTable * msg)
{
  if (!msg) {
    return false;
  }
  // table_size
  // ck_values
  if (!rosidl_runtime_c__double__Sequence__init(&msg->ck_values, 0)) {
    my_interfaces__msg__CkTable__fini(msg);
    return false;
  }
  // ck_values_average_in_range
  if (!rosidl_runtime_c__double__Sequence__init(&msg->ck_values_average_in_range, 0)) {
    my_interfaces__msg__CkTable__fini(msg);
    return false;
  }
  // total_erg_cost
  // total_erg_cost_in_range
  // erg_cost_reduction_perc
  // position
  if (!geometry_msgs__msg__Point__init(&msg->position)) {
    my_interfaces__msg__CkTable__fini(msg);
    return false;
  }
  return true;
}

void
my_interfaces__msg__CkTable__fini(my_interfaces__msg__CkTable * msg)
{
  if (!msg) {
    return;
  }
  // table_size
  // ck_values
  rosidl_runtime_c__double__Sequence__fini(&msg->ck_values);
  // ck_values_average_in_range
  rosidl_runtime_c__double__Sequence__fini(&msg->ck_values_average_in_range);
  // total_erg_cost
  // total_erg_cost_in_range
  // erg_cost_reduction_perc
  // position
  geometry_msgs__msg__Point__fini(&msg->position);
}

bool
my_interfaces__msg__CkTable__are_equal(const my_interfaces__msg__CkTable * lhs, const my_interfaces__msg__CkTable * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // table_size
  if (lhs->table_size != rhs->table_size) {
    return false;
  }
  // ck_values
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->ck_values), &(rhs->ck_values)))
  {
    return false;
  }
  // ck_values_average_in_range
  if (!rosidl_runtime_c__double__Sequence__are_equal(
      &(lhs->ck_values_average_in_range), &(rhs->ck_values_average_in_range)))
  {
    return false;
  }
  // total_erg_cost
  if (lhs->total_erg_cost != rhs->total_erg_cost) {
    return false;
  }
  // total_erg_cost_in_range
  if (lhs->total_erg_cost_in_range != rhs->total_erg_cost_in_range) {
    return false;
  }
  // erg_cost_reduction_perc
  if (lhs->erg_cost_reduction_perc != rhs->erg_cost_reduction_perc) {
    return false;
  }
  // position
  if (!geometry_msgs__msg__Point__are_equal(
      &(lhs->position), &(rhs->position)))
  {
    return false;
  }
  return true;
}

bool
my_interfaces__msg__CkTable__copy(
  const my_interfaces__msg__CkTable * input,
  my_interfaces__msg__CkTable * output)
{
  if (!input || !output) {
    return false;
  }
  // table_size
  output->table_size = input->table_size;
  // ck_values
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->ck_values), &(output->ck_values)))
  {
    return false;
  }
  // ck_values_average_in_range
  if (!rosidl_runtime_c__double__Sequence__copy(
      &(input->ck_values_average_in_range), &(output->ck_values_average_in_range)))
  {
    return false;
  }
  // total_erg_cost
  output->total_erg_cost = input->total_erg_cost;
  // total_erg_cost_in_range
  output->total_erg_cost_in_range = input->total_erg_cost_in_range;
  // erg_cost_reduction_perc
  output->erg_cost_reduction_perc = input->erg_cost_reduction_perc;
  // position
  if (!geometry_msgs__msg__Point__copy(
      &(input->position), &(output->position)))
  {
    return false;
  }
  return true;
}

my_interfaces__msg__CkTable *
my_interfaces__msg__CkTable__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  my_interfaces__msg__CkTable * msg = (my_interfaces__msg__CkTable *)allocator.allocate(sizeof(my_interfaces__msg__CkTable), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(my_interfaces__msg__CkTable));
  bool success = my_interfaces__msg__CkTable__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
my_interfaces__msg__CkTable__destroy(my_interfaces__msg__CkTable * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    my_interfaces__msg__CkTable__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
my_interfaces__msg__CkTable__Sequence__init(my_interfaces__msg__CkTable__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  my_interfaces__msg__CkTable * data = NULL;

  if (size) {
    data = (my_interfaces__msg__CkTable *)allocator.zero_allocate(size, sizeof(my_interfaces__msg__CkTable), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = my_interfaces__msg__CkTable__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        my_interfaces__msg__CkTable__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
my_interfaces__msg__CkTable__Sequence__fini(my_interfaces__msg__CkTable__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      my_interfaces__msg__CkTable__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

my_interfaces__msg__CkTable__Sequence *
my_interfaces__msg__CkTable__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  my_interfaces__msg__CkTable__Sequence * array = (my_interfaces__msg__CkTable__Sequence *)allocator.allocate(sizeof(my_interfaces__msg__CkTable__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = my_interfaces__msg__CkTable__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
my_interfaces__msg__CkTable__Sequence__destroy(my_interfaces__msg__CkTable__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    my_interfaces__msg__CkTable__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
my_interfaces__msg__CkTable__Sequence__are_equal(const my_interfaces__msg__CkTable__Sequence * lhs, const my_interfaces__msg__CkTable__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!my_interfaces__msg__CkTable__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
my_interfaces__msg__CkTable__Sequence__copy(
  const my_interfaces__msg__CkTable__Sequence * input,
  my_interfaces__msg__CkTable__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(my_interfaces__msg__CkTable);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    my_interfaces__msg__CkTable * data =
      (my_interfaces__msg__CkTable *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!my_interfaces__msg__CkTable__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          my_interfaces__msg__CkTable__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!my_interfaces__msg__CkTable__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
