# Sorting

Sorting can be done in many ways, some brute force, crude to really sophisticated ones. We start with Insertion sort, then move to Merge sort, then quick and Bucket sort.

## Bubble sort: Most basic way to sort

* iteratively place the largest element towards right or the largest index
* so, we start with an array and compare each element to its adjacent element, if greater shift towards right,
* this way in the first iteration we shift the largest element in the last index.
* we repeat this process until we have only one element in the left, that is the smallest element
* we start with the entire array and later the length of the array that we perform this comparisions keeps reducing until we are left with the smallest element.

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0,n-i-1):
            if arr[j]> arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                
    return arr
            
```

e.g. [64, 34, 25, 12, 22, 11, 90]

Visualization After Each Full Pass:

* After 1st Pass: [34, 25, 12, 22, 11, 64, 90]
* After 2nd Pass: [25, 12, 22, 11, 34, 64, 90]
* After 3rd Pass: [12, 22, 11, 25, 34, 64, 90]
* After 4th Pass: [12, 11, 22, 25, 34, 64, 90]
* After 5th Pass: [11, 12, 22, 25, 34, 64, 90]

Best Case Scenario: O(n)

Example for Best Case: [1, 2, 3, 4, 5]

First Pass:

* Compare 1 and 2, no swap needed.
* Compare 2 and 3, no swap needed.
* Compare 3 and 4, no swap needed.
* Compare 4 and 5, no swap needed.

No swaps were made, indicating the array is already sorted, and the algorithm can stop.

Worst Case Scenario: O(n^^2)

Example for Worst Case: [5, 4, 3, 2, 1]

First Pass:

* Swap 5 and 4: [4, 5, 3, 2, 1]
* Swap 5 and 3: [4, 3, 5, 2, 1]
* Swap 5 and 2: [4, 3, 2, 5, 1]
* Swap 5 and 1: [4, 3, 2, 1, 5]

Second Pass (and so on):
Continue with similar swaps until the array is sorted.

Space complexity is O(1) since its an inplace sorting algorithm

In this example, the algorithm has to perform the maximum number of passes (n-1) and swaps, making it highly inefficient due to the O(n^^2) time complexity.

## Selection sort

* While bubble sort finds the largest element and moves it to the right, selection sort finds the smallest element and moves it towards the left.

```python
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1,n):
            if arr[j]< arr[j+1]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
                
    return arr
            
```

Time Complexity:
Best case and worst case is both O(n^^2), since it has to do comparisions both the times.

Space Complexity : O(1) since its an inplace algorithm

Difference between bubble sort and selection sort:

we can say bubble sort, moves greatest things to the right, and selection sort moves smaller things to the left, but instead of swapping every adjacent element, it swaps only n-1 times via index.

Swaps or Memory Writes

Main Difference: The primary objective difference between Bubble Sort and Selection Sort lies in the number of swaps (or memory writes) they perform.

* Bubble Sort may perform up to O(n 2) swaps in the worst-case scenario.
* Selection Sort performs exactly n−1 swaps, regardless of the input array's initial order.

Time Complexity

* Worst-Case Complexity: Both algorithms share the same worst-case time complexity of O(n2) for comparisons. This means that in the worst case, both algorithms will have to perform a quadratic number of comparisons relative to the number of elements in the array.

* Best Case Complexity
  * Bubble Sort: With an optimization such as checking for an early stop (i.e., detecting a pass with no swaps), Bubble Sort can achieve a best-case time complexity of O(n) for nearly sorted or fully sorted arrays. This makes it adaptive to the initial order of the array.
  * Selection Sort: Lacks this adaptability regarding comparisons; it always performs O(n 2) comparisons because it does not adapt its behavior based on the sorted state of the array. Its performance benefit comes from minimizing swaps, not improving the best-case time complexity for comparisons.

Practical Implications

* Swaps/Write Operations: If minimizing the number of write operations is a priority (e.g., due to hardware constraints, large data records, or other efficiency concerns), Selection Sort may be preferred due to its lower number of swaps.
* Nearly Sorted Data: For nearly sorted data or when the dataset is small, an optimized Bubble Sort (with early stopping) might perform better due to its ability to finish early in such cases, leveraging its adaptive nature.

## Insertion sort

So the underlying assumption for the insertion sort is that we assume the first element is already sorted, and hence we insert each subsequent element into the correct position.

So, you keep moving across the array and for each iteration, you sort elements up until that point.

```python
def insertion_sort(arr):
    n = len(arr)
    for i in range(1,n):
        j = i-1
        while j>=0 and arr[j]> arr[j+1]:
            arr[j],arr[j+1] = arr[j+1], arr[j]
            j -=1

                
    return arr
            
```

There is an another way to do the same but instead of swapping, we would hold the current element that is key at index i, and we would shift elements before it, one position towards right and later would insert the key in the position where an element is no longer greater than key.

* The "key" is the element we are inserting each round.

* On paper, run through the key-based algorithm step-by-step. For example: Array [5, 2, 4, 1, 3]. Take key=2. Compare to 5, shift 5 ahead. Insert 2 before 5.

```python
def insertion_sort(arr):
    n = len(arr)
    for i in range(1,n):
        j = i-1
        key = arr[i]
        # Move elements of arr[0..i-1], that are 
        # greater than key, to one position ahead 
        # of their current position 
        while j>=0 and arr[j]> key:
            arr[j+1] = arr[j]
            j -=1
        # Place the key at after the element just smaller than it.
        arr[j+1] = key

                
    return arr
            
```

So in summary:

* Selection sort Selects -> Directly places
* Insertion sort Inserts -> Opens up space and inserts at position

A swap operation is generally more expensive than a shift operation in most programming languages:

* More assignments: A swap involves 3 assignments - copying one value into a temporary variable, then assigning second value to first, and temporary to second. Shift just assigns second value to first variable.
* Additional temporary variable: A swap operation needs a temporary variable to store one of the values before overriding it. This additional memory access/variable increases swap overhead.
* More cache invalidations: When variables change values, it causes more data to become invalid in processor caches. A swap's 3 changes invalidate more cached data than a shift's single change.
* Context switches: In multithreaded environments, the multiple swap assignments have greater chance of causing CPU context switches between threads.

Time Complexity
Best Case: O(n) - When the array is already sorted, each key only needs to be compared with its predecessor, resulting in minimal operations.

Average and Worst Case: O(n 2) - When the array is sorted in reverse order, each element needs to be placed at the beginning of the sorted portion, requiring a significant number of comparisons and shifts.

Space Complexity

Space Complexity:O(1) - Insertion Sort is an in-place algorithm, meaning it only requires a constant amount of additional space for its operations.

When to Use Insertion Sort

* Small Datasets: It is highly efficient for small to medium-sized datasets due to its low overhead.

* Nearly Sorted Datasets: Insertion Sort is particularly effective for datasets that are already nearly sorted, as it minimizes the number of necessary operations.

* Online Sorting: Suitable for scenarios where data is received incrementally and needs to be sorted in real-time.

**Suitable for Online Sorting**

* Adaptability: Algorithms that can efficiently insert new elements into an already sorted sequence without needing to re-sort the entire dataset from scratch are well-suited for online sorting. This adaptability minimizes computational overhead with each new insertion.

* In-place Sorting: Algorithms that sort the data in place, without requiring significant additional memory, are preferable since they can handle data as it arrives without needing a lot of extra space.

* Stability: In many online sorting scenarios, maintaining the relative order of equivalent elements can be important, making stable algorithms (which do not change the relative order of elements with equal keys) more suitable.

Examples:

* Insertion Sort is particularly well-suited for online sorting because it can efficiently add new elements into the correct position within an already sorted array.

* Heap Sort can be adapted for online use by maintaining a heap data structure. As new elements arrive, they can be added to the heap, and the heap can be rebalanced efficiently.

* Merge Sort and Quick Sort are **typically not used** for online sorting in their standard forms due to their need for the entire dataset to determine how to split or pivot the data. 
* While highly efficient for batch sorting, adapting them for online sorting can be complex and may negate their batch processing advantages.
* Radix Sort and Counting Sort, while efficient for specific data types and ranges, also generally require knowledge of the entire dataset to perform effectively and are thus not ideal for online sorting scenarios.

## Merge Sort: Sorting by divide and conquer

Divide the array possibly into equal halves if it is of even length.

```python
def merge_sort(arr):
    n = len(arr)
    if n <= 1:
        return arr
    L = arr[:n//2]
    R = arr[n//2 : ]

    merge_sort(L)
    merge_sort(R)
    merge(L,R,arr)

def merge(L,R, arr):
    i=j=k = 0

    while i < len(L) and j < len(R):
        if L[i] <= R[j] :
            arr[k] = L[i]
            i +=1
            k +=1
        else:
            arr[k] = R[j]
            i += 1
            k += 1

    if i < len(L):
        arr.extend(L[i:])
    if j < len(R):
        arr.extend(R[j:])
            
```
Notes on recursion : 

(Taken from a comment on Khan academy at https://www.khanacademy.org/computing/computer-science/algorithms/merge-sort/a/overview-of-merge-sort) 

When you use recursion, there may be several copies of a function, all at different stages in their execution. When one function returns the function that called it continues to execute.

- I'll denote each step that each version of the function is executing with a *.



Suppose we call MergeSort on [4,3,2]

The 1st copy of the function will be made which looks like this:
>  * MergeSort([4,3]) <br>
   MergeSort([2]) <br>
   Merge Above Together <br>


The call to MergeSort([4,3]) will then generate a 2nd copy of the function
(shown below the 1st copy)

>  *MergeSort([4,3]) <br>
   MergeSort([2]) <br>
   Merge Above Together <br>

>  *MergeSort([4]) <br>
   MergeSort([3]) <br>
   Merge Above Together <br>


The call to MergeSort([4]) will then generate a 3rd copy of the function
(shown below the 2nd copy)

>  *MergeSort([4,3]) <br>
   MergeSort([2]) <br>
   Merge Above Together <br>

>  *MergeSort([4]) <br>
   MergeSort([3]) <br>
   Merge Above Together <br>

>  *Do Nothing <br>


The Do Nothing step will finish. The 3rd copy of the function will return (and vanish).
The 2nd copy of the function will move on to the next line.

>  *MergeSort([4,3]) <br>
   MergeSort([2]) <br>
   Merge Above Together <br>

>  MergeSort([4]) <br>
  *MergeSort([3]) <br>
   Merge Above Together <br>


The call to MergeSort([3]) will then generate a 3rd copy of the function
(shown below the 2nd copy)

>  *MergeSort([4,3]) <br>
   MergeSort([2]) <br>
   Merge Above Together <br>

>   MergeSort([4]) <br>
  *MergeSort([3]) <br>
   Merge Above Together <br>

>  *Do Nothing <br>


The Do Nothing step will finish. The 3rd copy of the function will return (and vanish).
The 2nd copy of the function will move on to the next line.
>  *MergeSort([4,3]) <br>
   MergeSort([2]) <br>
   Merge Above Together <br>

>   MergeSort([4]) <br>
   MergeSort([3]) <br>
  *Merge Above Together <br>


The Merge Above Together step will finish. The 2nd copy of the function will return (and vanish).
The 1st copy of the function will move on to the next line.
>   MergeSort([4,3]) <br>
  *MergeSort([2]) <br>
   Merge Above Together <br>

The call to MergeSort([2]) will then generate a 2nd copy of the function
(shown below the 1st copy)
>  MergeSort([4,3]) <br>
  *MergeSort([2]) <br>
   Merge Above Together <br>

> *Do Nothing


The Do Nothing step will finish. The 2nd copy of the function will return (and vanish).
The 1st copy of the function will move on to the next line.

>  MergeSort([4,3]) <br>
   MergeSort([2]) <br>
   *Merge Above Together <br>

The Merge Above Together step will finish. The 1st copy of the function will return (and vanish).
The array should now be sorted.

Now lets discuss the complexity

Two things are happening here, 
at any function call, if an arr is greater than the length of 1, we divide the array further into halves and sort them and merge them.

Consider both L and R at each level, at each level we have recursive call and a merge function call. 
* At each call, or a recursive level, once both the L and R functions returns, we will have n elements to compare and merge across the tree level. So the merge function complexity is O(n)
* Given that, we now have to determine how many levels will be in the recursion tree, consider L and R on both the sides as one level and each of them will further be divided into L and R. 
And the level stops when each L and R finally reach the size of 1.

e.g. if orignial array length is 8. How many times will there be a copy of the original function via recursion, which is our depth of the recursion depth.
* At first level, it will be divided into L(4) and R(4)
* At second level, each L(4), R(4) will be divided into L(2) and R(2)
* At third level, each L(2), R(2) will be divided into L(1) and R(1) and now it stops.

so, it takes 3 levels or recursion tree depth = 3, for an array of 8.  

3 = log 8

So, the no. of level is log n and at each level and the merge function will take n steps at any level to compare, so we the complexity is n log n

Please note that it is recursion and not all the levels are being executed once and we need to remember or have space only for n elements.

Space Complexity: At any level, we are storing the all the elements and it is O(n) and we also need to store function calls on the stack space, so O(log n) + O(n), which is equivalent to O(n)

Recursion Stack space:

* In programming languages, function calls are stored in a data structure called the call stack
* When any function is called, an entry is "pushed" onto this call stack containing info like arguments, local variables etc.
* When the function exits/returns, that entry is "popped" off the stack.

For recursive functions:

* On each recursive call, a new entry is added to the call stack.
So if we have a recursion tree of height 'h' - it means the call stack will reach a maximum depth of 'h' entries.

## Quick Sort

### Origins

* Invented in 1960s by Tony Hoare while in graduate school
* At the time, merge sort (1945) was an established efficient sorting technique with O(n log n) performance. But high memory overhead O(n) limited hardware capabilities back then. We needed something inplace
* Hoare devised an alternative divide-and-conquer approach without actual merging step to achieve similar efficiency without excessive memory load

```python
def quicksort(arr):
    if len(arr) <= 1:  # Explicit base case for recursion
        return arr
    else:
        pivot = arr[0]

        left = [x for x in arr[1:] if x < pivot]   
        middle = [x for x in arr if x == pivot]  # Ensure all duplicates of pivot are included
        right = [x for x in arr[1:] if x >= pivot]

        return quicksort(left) + middle + quicksort(right)
            
```

While elegant, this method creates new lists for left, middle, and right in each recursive call, a total of O(n), which can lead to higher memory usage compared to in-place partitioning strategies, where you get memory advantage relative to Merge Sort.

Use BFS When:

* Finding the Shortest Path: BFS is ideal for finding the shortest path in unweighted graphs because it explores all nodes at the present depth before moving on to nodes at the next depth level. This characteristic ensures that the first time a node is reached, it is by the shortest possible route.
Example: In a maze or grid, using BFS to find the shortest route from one point to another.

* Level Order Traversal:
If you need to traverse a tree or graph level by level (e.g., level order traversal in trees), BFS naturally processes elements in this manner using a queue.
Example: Printing nodes of a tree in level order.

* Spreading Processes:For problems modeling processes that spread or propagate from multiple sources simultaneously, such as in network broadcasting or infection spreading where each step of the process needs to be tracked globally.
Example: Calculating the minimum time required for all oranges to rot when some are initially rotten.

Use DFS When:

* Checking Connectivity or Component Count: DFS is often simpler to implement recursively and is useful for exploring all nodes in a connected component. It's great for problems where you need to explore as much as possible from each node once you start from it.
Example: Counting the number of connected components in a graph or grid.

* Finding Cycles: DFS can be more effective in cycle detection in directed and undirected graphs because it explores paths deeply before backtracking.
Example: Detecting cycles in a graph to check if it’s possible to complete all courses given prerequisite relations.

* Path Finding with Constraints: If the problem involves exploring paths or combinations with certain constraints where not all paths are viable, DFS can effectively use backtracking to explore possible solutions deeply and backtrack on dead ends.
Example: Solving puzzles like Sudoku or searching for a path that meets specific criteria.

* Stack Space and Recursion: When a solution can be expressed as a recursion, DFS can be directly applied using the system stack. This makes it easier to write for deep exploration tasks.
Example: Deep search in file systems or complex nested structures.

General Guidelines:

* BFS is typically used for shortest path problems or when you need to explore all options uniformly. It requires more memory than DFS as it stores all nodes of the current depth level to process the next level.
* DFS uses less memory and can be more efficient for exploring all possible ways from a given node if the path does not need to be the shortest. However, it can get stuck in deep paths or recursion limits in large graphs.

Choosing Based on the Problem:

* Evaluate the problem’s requirements:

* If the problem asks for the shortest path or level-wise processing, BFS is likely the right choice.
* If the problem involves exploring configurations, solving puzzles, or needs deep search capabilities, DFS might be more suitable.
* Ultimately, the specific problem constraints and requirements will guide which traversal method is best suited for the task.