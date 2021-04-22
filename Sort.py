##################################################
## Sorting Algorithms and Applications
##################################################
##################################################
## Author: Alexander Railton
## Copyright: Copyright 2021, Sorting
## Credits: Alexander Railton
## Number: (416)-951-8621
## Email: alexanderraiton@gmail.com
## Status: {dev_status}
##################################################

#fucntions
def merge_sort(array, left_index, right_index):
    if left_index >= right_index:
        return

    middle = (left_index + right_index)//2
    merge_sort(array, left_index, middle)
    merge_sort(array, middle + 1, right_index)
    merge(array, left_index, right_index, middle)

def merge(array, left_index, right_index, middle):

    # make copies of both arrays
    left_copy = array[left_index, midlle+1]
    right_copy = array[middle+1, right_index+1]

    # Initialize values for variables that we use
    # to track values that we use track in each 
    # array.
    left_copy_index = 0
    right_copy_index = 0 
    sorted_index = left_index

    # Go through both coppies until we run out of elements
    while left_copy_index < len(left_copy) and right_copy_index < len(right_copy):
        # if our left_copy has a smaller element, put it in the sorted
        # then move to the left_copy accordingly
        if left_copy[left_copy_index] <= right_copy[rihgt_copy_index]:
            array[sortedindex] = left_copy [left_copy_index]
            left_copy_index = left_copy_index + 1
        else:
            array[sorted_index] = right_copy[right_copy_index]
            right_copy_index = right_copy_index + 1

    # Regardless of where we got our element from
    # move to the second part
    sorted_index = sorted_index + 1

    # We ran out of elements either in left_copy or _right_copy
    # so we will go through the remaining elements and add them on
    while left_copy_index < len(left_copy):
        array[sorted_index] = left_copy[left_copy_index]
        left_copy_index = left_copy_index + 1
        sorted_index = sorted_index + 1

    while right_copy_index < len(righ_copy):
        array[sorted_index] = right_copy[right_copy_index]
        right_copy_index = right_copy_index + 1
        sorted_index = sorted_index + 1


def merge(array, left_index, right_index, middle):
    # Make copies of both arrays we're trying to merge

    # The second parameter is non-inclusive, so we have to increase by 1
    left_copy = array[left_index:middle + 1]
    right_copy = array[middle+1:right_index+1]

    # Initial values for variables that we use to keep
    # track of where we are in each array
    left_copy_index = 0
    right_copy_index = 0
    sorted_index = left_index

    # Go through both copies until we run out of elements in one
    while left_copy_index < len(left_copy) and right_copy_index < len(right_copy):

        # If our left_copy has the smaller element, put it in the sorted
        # part and then move forward in left_copy (by increasing the pointer)
        if left_copy[left_copy_index] <= right_copy[right_copy_index]:
            array[sorted_index] = left_copy[left_copy_index]
            left_copy_index = left_copy_index + 1
        # Opposite from above
        else:
            array[sorted_index] = right_copy[right_copy_index]
            right_copy_index = right_copy_index + 1

        # Regardless of where we got our element from
        # move forward in the sorted part
        sorted_index = sorted_index + 1

    # We ran out of elements either in left_copy or right_copy
    # so we will go through the remaining elements and add them
    while left_copy_index < len(left_copy):
        array[sorted_index] = left_copy[left_copy_index]
        left_copy_index = left_copy_index + 1
        sorted_index = sorted_index + 1

    while right_copy_index < len(right_copy):
        array[sorted_index] = right_copy[right_copy_index]
        right_copy_index = right_copy_index + 1
        sorted_index = sorted_index + 1

# main
# array = [33, 42, 9, 37, 8, 47, 5, 29, 49, 31, 4, 48, 16, 22, 26]
# merge_sort(array, 0, len(array) -1)
# print(array)

# Sorting Objects
##################################################################
class Car:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year

    def __str__(self):
        return str.format("Make: {}, Model: {}, Year: {}", self.make, self.model, self.year)

#functions
def merge(array, left_index, right_index, middle, comparison_function):
    left_copy = array[left_index:middle + 1]
    right_copy = array[middle+1:right_index+1]

    left_copy_index = 0
    right_copy_index = 0
    sorted_index = left_index

    while left_copy_index < len(left_copy) and right_copy_index < len(right_copy):

        # We use the comparison_function instead of a simple comparison operator
        if comparison_function(left_copy[left_copy_index], right_copy[right_copy_index]):
            array[sorted_index] = left_copy[left_copy_index]
            left_copy_index = left_copy_index + 1
        else:
            array[sorted_index] = right_copy[right_copy_index]
            right_copy_index = right_copy_index + 1

        sorted_index = sorted_index + 1

    while left_copy_index < len(left_copy):
        array[sorted_index] = left_copy[left_copy_index]
        left_copy_index = left_copy_index + 1
        sorted_index = sorted_index + 1

    while right_copy_index < len(right_copy):
        array[sorted_index] = right_copy[right_copy_index]
        right_copy_index = right_copy_index + 1
        sorted_index = sorted_index + 1


def merge_sort(array, left_index, right_index, comparison_function):
    if left_index >= right_index:
        return

    middle = (left_index + right_index)//2
    merge_sort(array, left_index, middle, comparison_function)
    merge_sort(array, middle + 1, right_index, comparison_function)
    merge(array, left_index, right_index, middle, comparison_function)

# Object Merge Sort Driver
car1 = Car("Alfa Romeo", "33 SportWagon", 1988)
car2 = Car("Chevrolet", "Cruze Hatchback", 2011)
car3 = Car("Corvette", "C6 Couple", 2004)
car4 = Car("Cadillac", "Seville Sedan", 1995)

array = [car1, car2, car3, car4]

merge_sort(array, 0, len(array) -1, lambda carA, carB: carA.year < carB.year)

print("Cars sorted by year:")
for car in array:
    print(car)

print()
merge_sort(array, 0, len(array) -1, lambda carA, carB: carA.make < carB.make)
print("Cars sorted by make:")
for car in array:
    print(car)

# Optimizing sorting methods
# to do a bottom - up method
# With real life data we often have a lot of these 
# already sorted subarrays that can noticeably
# shorten the execution time of Merge Sort.

# Another thing to consider with Merge Sort,
# particularly the top-down version is multi-threading.
# Merge Sort is convenient for this since
# each half can be sorted independently of its pair.
# The only thing that we need to make sure of
# is that we're done sorting each half
# before we merge them.

# Merge Sort is however relatively inefficient
# (both time and space) when it comes to smaller
# arrays, and is often optimized by 
# stopping when we reach an array of ~7 elements,
# instead of going down to arrays with
# one element, and calling Insertion Sort
# to sort them instead, before merging
# into a larger array.