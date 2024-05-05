# bubble sort:

1 2 5 4 6

5 1 7 8 2



def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0,n-i-1):
            if arr[j] > arr[j+1]:
                arr[j],arr[j+1] = arr[j+1],arr[j]
        return arr
    
# but bubble sort may sometimes be swapping unnecessarily, although doesn't help much with
# the time complexity, we can look into reducing swaps by only swapping elements already in place.
# se can look for the smallest element and place it towards the left, can be done in other way as well.
       
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1,n):
            if arr[j]< arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

# however, this still doesnot exploit the structure in the array.
# for an array of  n elements, it always does n-1 comparisionss, then n-2,  and so on. n(n-1)/2
# but lets say we have a sorted array, we need to compare few elements until the pivot point.
# so, if its a sorted array, all good otherwise, we can construct such structure,
# by placing each element into array.

def insertion_sort(arr):
    n = len(arr)
    for i in range(1,n):
        key = arr[i]
        j = i-1
        while j>0  and arr[j]> arr[key]: # with this, you avoid comparisions in the sorted set
            arr[j+1] = arr[j] #only if the element is less than the current element, we shift right to make space for the current element
            j -= 1     # so in the best case, when the array is almost sorted, we need to do only n-1 comparisons.
        arr[j+1] = key # in the worst case like a reverse sorted array, it still needs O(n2)
    return arr

# given O(n2), we have to resort to approaches like divide and conquer and first comees Merge Sort
# You divide the array into two halves, sort them each separately, by dividing each of these subsets
# into further smaller sets until there is only 1 element, then we merge them sorted sub array back together
# however, it requires a space complexity of the order of input array, 

def merge_sort(arr):
    n= len(arr)
    if n <= 1:
        return arr
    L = arr[:n//2]
    R = arr[n//2:]
    merge_sort(L)
    merge_sort(R)
    merge(L,R,arr)

def merge(L,R,arr):
    i = j = k = 0

    while i < len(L) and j < len(R):
        if L[i] <= R[j]:
            arr[k] = L[i]
            i +=1
        else:
            arr[k] = R[j]    
            j +=1

        k +=1

    if i < len(L):
        arr.extend(L[i:])
    if j < len(R):
        arr.extend(R[j:])


# this requires space complexity for O(n) for storing L and R across each tree level.

# in merge sort, in the divide step, we just divide and nothing related to sorting happends
# and all the work happens in the divide step, however it requires additional space

# what if instead of comparing each element with each other, we just take a single element
# and place it in its right position in the array. so, we take a pivot element 
# and place all the elements less than it irrespective of its order in the left side
# and the greater than elements to its right side.
# then we break the array into two parts left and right portions without pivot,
# select a new pivot for each of them and do the same.
# we do it until the left and right are left with less than 2 elements.
        
def quicksort(arr, low, high):
    if low < high:
        pivot_index = partition(arr, low,high)
        quicksort(arr,low, pivot_index-1)
        quicksort(arr,pivot_index+1, high)

def partition(arr, low,high):
    # the objective is to place a pivot in a correct position right
    # so we chose our pivot, lets say last one on the RHS and then what ?
    # we compare each element with our pivot, kets track them by j runs from 0 to n
    # and also need a pointer where we have to place our pivot element,
    # like if i indicates the place where there are greater elements than pivot 
    # we place pivot at i-1 
    # lets start i with the starting 0, 
    # coz once we find an element less than pivot, we want to place that at 0 and 
    # our indication pointer moves to 1, so that we can place the pivot next to it.

    # return the pointer

    pivot = arr[high]
    i = low # pointer for thr greater element, lets start from the left side and iterate towards right
    for j in range(low,high):
        if arr[j]<= pivot:
            
            arr[i], arr[j] = arr[j], arr[i]
            i +=1
    # place the pivot where you no longer see elements less than pivot
    arr[i],arr[high] = pivot, arr[i]
    return i

# the time complexity if the pivot is not selected such that the partition doesnot 
# happen in middle or something, then the length of the tree becomes n, 
# like 1 element in the right side, and all the others in the left side, 
# this happens if the array is already sorted. 
# and for each level, we are doing n comparisions and hence can result in O(n2).
# otherwise, its more or less be O(n log n) like mergesort and it doesn't need any space O(n) like merge sort

## HEAP SORT
# We can use a binary tree data structure from the given array and then heapify it by making
# sure that the parent node is always greater than the child nodes
# if we do so, we arrive at a structure, where we have maximum value at the top
# then we extract this maximum value and swap it with the last element in the array.
# FYI, we are not constructing a different data structure for creating a binary tree of the elements.
# we need to use the same array for space optimization, also array gives easy access via indexing at the top/root index
# So, once we extract the maximum value, and swap it with the last element, we place the maximum value
# at the last position in the array, as it should be. Thus, we sort one element.
# now, we re heapify the array, but this time apply heapifization only to the top nodes and
# so that the maximum element is not affected again. Then we extract maximum element and keep doing it

# heapifying just implies having smallest or largest object always at the top

# for an array to be represented as a binary tree, the convention is that
# a node at i has its children, left node at 2*i+1, right node at 2*i -1
# similarly a node at i has its parent node at, (i-1)//2, 
# e.g. node at 3 has parent node at 1 , (3-1)//2
# e.g. node at 4 also has parent node at 1, (4-1)//2 

# and for an array of n elements, there will be how many non leaf nodes or non terminal nodes
# we care about non leaf nodes, beacuse we want to apply max heap on these nodes or on these functions
# for 2-3 elements, 1 non leaf leaf node
# for 4-7 elements, 2-3 non leaf nodes
# for 8-13 elements, 4-7 non leaf nodes
# for an array of 8 elements, so the last non leaf node would be 8//2 = 4th element and its index is 3
## so, non leaf node for n elements would be n//2 - 1 

def heap_sort(arr):
    n = len(arr)

    # build the max heap from the input array and start extracting the max elements and heapifying it again

    # build the max heap array, i.e., for each of the non leaf node nodes, 
    # non leaf positions in the array, make sure that the parent node is greater than the child nodes

    for i in range(n//2-1,0,-1):
        heapify(arr,n,i)


    for i in range(n-1,0,-1):
        # extract max element and place it at the last index
        arr[0],arr[i] = arr[i], arr[0]
        # heapify again the array part excluding the last element, 
        # that implies the array length or heap size is reduced by 1
        # Earlier, it was n, now it becomes i, which is i-1 
        heapify(arr,i, 0)

        # now lets write the heapify function

# for heapifying an a
def heapify (arr, n, i):

    # lets assume the largest element be at i, if not we shift it
    largest = i
    left_element = 2*i + 1 
    right_element = 2*i + 1

    # check if a left element exists, if so, check if its greater than the left element
    # if so, swap it and lets apply the heapify again on the child just to make sure

    # if an array has on 4 elements, it implies it doesnot have right node for second element,
    # and no child nodes for the third element. Similarly, for left node
    # hence we just need to check for a given node, i, and its left and right nodes,
    # we need to check if the array of n has those elements

    # also in case of heapifying the sub trees n decides the reduced tree range
    # we do not want to include whole tree, when we already sorted few large elements to the 
    # end of the array
    if left_element < n and arr[left_element] > largest:
        largest = left_element
    if right_element < n and arr[right_element] > largest:
        largest = right_element

    # so we now know the index of thelargest among all the three nodes
    # so, if we find that i is no longer largest, we swap
    
    if i != largest:
        arr[i], arr[largest] = arr[largest],arr[i]
        # since we swapped, we need to check if the swap violated any max heap in case child node has any further nodes
        # if we are here, it implies largest node refers to the child node
        # and we need to check max heap at that node; largest
        heapify(arr,n,largest)

# in a binary tree, the height of the tree for n elements is logn
# the maximum comparision performed in single call of heapify is equal to the height of the tree
# for 8 elements, for a single heapify, we do a max comparion for entire tree = 3 times for left and right
        # single heapify call is logn

# for creating max heap from the array, we call heapify n//2 times 
# so total till now is nlogn
        
# however, deeper levels contribute less to the complexity and its complexity for a tree of height h is linear O(h)
# like for lowest level, which contains half nodes, we don't do any comparison.
# for 1st level nodes, we do at the max 1 swap and 1 layer of comparision hence one level height and not logn heigt
# as, we move to the top that we might do a comparision which covers the entire tree, hence logn 
# when you aggregate it together, it all cancels out and we only do O(n) operations.        

# however, after building max heap and for sorting, we remove the top element, and perform heapify n times
# and this heapify function has higher chances of covering the entire tree length, as we are removing the largest element
# and replacing it with some smallest element, hence logn for each heapify call and hence a total of nlogn


# and after for extracting we are heapifying again so, n times* heapify complexity = O(n* logn)
# so, O(n)+ O(nlogn) ~ O(nlogn)
# space complexity is 1 or constant as we do inplace sorting.

####
### but what if we are dealing with categorical elements like strings or integers, can we do sorting on it

### Counting Sort : lets say we have an array has elements within a range,
### then we can create a counting array and count each of the occurence of elements within a range
## and populate the array back

def countingsort(arr):
    n =len(arr)
    max_element = max(arr)
    count = [0]*(max_element + 1) # +1 as we need to have a place for counting 0's as well
    output_array = [0]*n

    for num in arr:
        count[num] += 1 # count each number

    i = 0
    for n in range(len(count)):
        for j in range(count(n)):
            arr[i] = n
            i += 1 
    return arr

# however this doesn't maintain the relative ordering of the duplicate elements
# lets say you have [10,1,9,2,7,1,4,1]
# we are just replacing the elements in the main array without relative ordering of 1s.

# so, to maintain the relative order, when placing 1's in the array, 
# we need to place last 1 to the last of 1s, i.e., at index 2
# how do we do that, we can start from the reverse, like starting last 1 when placing into array
# to maintain the relative ordering of 1s. 
# ok, once we place 1, we need to know how many 1's are still left, to place the next occuring 1 appropriately at the right position
# we can go by just count and place it at 0. However this method doesnot work for example like below and 
# requires cumulative sum for knowing how many elements less or equal to are present before it.


# otherwise

# lets say you have 2 2's and 1 1's and the rest of the integers below 20 in an array.

# array = [13, 1, 18, 2,5,10, 2]

# lets say we start from last 2 and we know the count is 2 and we place it at index 2- 1 = , and reduce count to 1. now we move to another 2 and place it at first 1-1 = 0. Now what happens to placing 1, we know the count of 1 is 1 and we need to place it at 1-1 = 0 index, but there is no place.

# But if we go by cumulative sum, we know how many elements are before it and we can place it appropriately.

def counting_sort(arr):
    n = len(arr)
    maximum = max(arr)
    count = [0]*(maximum+1)
    output_array = [0]*len(arr)

    # find the count 
    for num in arr:
        count[num] +=1

    # palcing numbers from the reverse, helps maintain the relative ordering for duplicate elemets
    # however, we need cumulative sum to have an idea on how many elements are there <= element to be placed
    # so that it can be placed appropriately, without overwriting and
    # maintaining enough space for elements before it
        
    for i in range(1, len(count)):
        count[i] += count[i-1]      # take cumulative sum, this helps in finding the right index
    
    for i in range(n-1,-1,-1):
        output_array[count[arr[i]]-1] = arr[i] 
        count[arr[i]] -= 1
    
    for i in range(n):
        arr[i] = output_array[i]
    
    return arr

# we go through each element in arr once for counting n
# we go through count array, lets say k for an array of 10 elements between 1 to 5, count array can only be of size 5
# the max time complexity can be O(n+k), the space complexity is also the same.

# but counting sort becomes unncessarily expensive if an array has elements lets ranging from 1 to 100000
# you create a count array for each of the element in the range

# there should be a better way to do it and its radix sort
# for each exponent place like 1's , 10's and 100's digits we sort the elements and starting from 1's to max exponent
# then the numbers get sorted at the end
# we can use counting sort, as we would have idea on the range like 0 to 9

# we only need to find max exponent and do the sorting those many times

def radix_sort(arr):
    max_num = max(arr)
    exp = 1
    while max_num // exp > 0:
        counting_sort(arr,exp)
        exp *= 10
    return arr

def counting_sort(arr,exp):

    n = len(arr)
    count = [0]*10
    output_array = [0]*n

    for i in range(n):
        # for a number 425, if we are currently sorting by 10's place like 2
        # how do we get it, of course you can convert to string and get the second last element
        # otherwise, 2 is 10's place, for that we need to remove both 5 and 4 i.e., 400
        # 5 can be removed by doing only integer division without rounding like 425// 10 ~ 42
        # for removing left element i.e., 4, then we need remainder 2 42%10 ~ 2

        index = (arr[i] // exp) % 10 
        count[index] +=1

    for i in range(1,len(count)):
        count[i] += count[i-1]

    for i in range(n-1,-1,-1):
        index = (arr[i] // exp) % 10 
        output_array[count[index]-1] = arr[i]
        count[index] -= 1

    for i in n:
        arr[i] = output_array[i]


# best case time complexity is O(d*(n+k)) where d is the number of digits.
# worst case would be when all the elements do not have same number of digits, 
# if the largest element has n digits, the complexity can become O(n2)
# space complexity = O(n+k)

# quick select
# to find the kth largest element in an array
# we need not do the sorting of the entire array
# we just need to sort the array until kth element, or (n-k)th element
# so we need to sort only in one direction like keep finding the largest k elements and return kth element
        
# we can do like max heap: build max heap O(n) and extract only k element O(klogn)
# else, go quick sort way, like consider a pivot, sort it and find its position, if its k-1 from the right side
# then return it

# if we recursively sort only one side of the array, then the time complexity is logn, on average its O(n)

def quickselect():
    
    return

def quicksort(arr,low, high):
    pivot = partition(arr,low,high)
    quicksort(arr,low,pivot-1)
    quicksort(arr,pivot+1,high)

def partition(arr,low,high):
    pivot = arr[high]
    
    i = low # where to place the pivot
    for j in range(low, high):
        if arr[j] <= pivot :
            arr[j], arr[i] = arr[i], arr[j]
            i+= 1
    
    arr[i], arr[high] = pivot, arr[i]
    return i


def quickselect(arr, low, high,k):
    if low == high:
        return arr[low]
    pivot = partition(arr,low,high)
    if pivot == len(arr) - k :
        return arr[pivot]
    elif pivot < len(arr) - k: # pivot is in the left side
        return quickselect(arr,low, pivot-1,k)
    else: 
        return quickselect(arr, pivot+1, high, k)



# ok, here, if k is large and is closer to n and is also nearly sorted.
# if we do quick sort, and if we select the last element as the pivot, 
# most of it reminas the same and no swapping occurs, as it is nearly sorted.
# so, we isolate the pivot and apply quicksort on the left over array.
# now we do the same again with the last elememt chosen as the pivot.
# so, for each recursion call, we are only reducing the order by 1 element, instead of halving.
# thus it might result in n(n+1)/2 time complexity.

# if k is large and it might be better to sort the entire array or use a version of min heap


## else, we can use max/min heap structures, to arrange the array in the form of binary tree
## for max heap, you contruct the binary tree and remove the max element k times to get the kth largest element
## building max heap once takes O(n) complexity

# else, we can go for min heap, where you have minimum element at the top.
# construct a min heap of k size, where root is smaller than all the k-1 elements
# now go through the array, and if the next element is greater than root, you remove the root and add the new element, and min heapify it again
# thus now you have k elements of which root element is the smallest.
# if we continue to next and throughout the array, we will have an array, which has k largest elements and the root element is the smallest.
# space complexity : k
# time complexity: (n-k) logk, as we need to call min heapify n-k times after comparision, and each heapify takes logk time.

# by considering only k, we have only (n-k) logk, which otherwise would have been O(n) + klogn

import heapq
def kth_largest_element(nums,k):
    # lets use heapq which constructs min heap
    min_heap = nums[:k]
    heapq.heapify(min_heap)

    for i in range(k, len(nums)):
        if nums[i]> min_heap[0]:
            heapq.heappop(min_heap)
            heapq.heappush(min_heap,nums[i])
    
    return min_heap[0]

class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        """
        counts = [0]*3
        for num in nums:
            counts[num] +=1
        #print(counts)
        k = 0
        for i in range(len(counts)):
            for j in range(k,k+counts[i]):
                nums[j] = i
            #print(nums)
            k += counts[i]
        ## however, this takes two passes, one for counting, one for asisgnment
        ## time complexity O(2n), space complexity = len(counts)

        """
        ## what if we use three pointers (multiple pointers are mostly used for single pass)
        ## we use 3 pointers, one for red, which indicates the index below which all should be red 0
        ## starts from 0

        ## one for blue, indicates above which all should be blue, starts from the end
        ## the middle one for while, which we use just to evaluate the cuurent element
        ## if its equal to 1, we increment it by 1
        ## if the current element is 0, we swap it with the element at the red pointer and increment both the red and white pointer
        ## otherwise, if its blue, we place it at the end and and decrement the blue pointer by 1 towards left
        ## thus within a single pass, we iterate through the array, while segregating it into 3 parts.
        red, white, blue = 0, 0, len(nums)-1
        
        while white <= blue:
            if nums[white] == 0:
                nums[red], nums[white] = nums[white], nums[red]
                red += 1
                white += 1
            elif nums[white] == 1:
                white += 1
            else:
                nums[white], nums[blue] = nums[blue], nums[white]
                blue -= 1
        





### BINARY SEARCH
# we can search for an element in array in O(n)
# else, assuming an array is sorted, we can also search in only one direction like in quick select
# resulting in O(logn) time and constant space complexity O(1) for constants like left and right pointers
                
def binary_search(arr,target):
    l, r = 0, len(arr)-1

    while l<=r:
        mid = (l+r)//2 ## floor division
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            l = mid+1
        else:
            r = mid-1

    return -1

def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m = len(matrix)
        n = len(matrix[0])
        """
        flag = False
        for i in range(m):
            arr = matrix[i]
            l = 0
            r = len(arr)-1
            while l<=r:
                mid = (l+r)//2
                if arr[mid] == target:
                    flag = True
                    return True
                elif arr[mid] < target:
                    l = mid + 1
                else:
                    r = mid -1
        return flag
        """
        
        # however the above returns false, as soon as it finds nothing in the first row, without moving to the next row.
        # hence converted to flag.  
        # it operates with the time complexity of  O(m*logn), binary search over n elements for m rows
        
        # however it still doesn't use the structure of the array, i.e. last element of 1st row is lower than the 1st element of the second row.
        # it implies that its already a total sorted array.
        # without directly flattening, we can represent any index lets say 3 in a (3,3) matrix  is (1,0)
        # for idx 3, row = 1 = 3//3 = id//n where n is the number of columns
        # for idx 3, column = 0 = 3% 3 = idx %n, where n is the number of columns
        # with this, we do single binary search directly over m*n elements, hence the time complexity O(log(m*n))

        l,r = 0, m*n - 1
        while l <=r:
            mid = (l+r)//2
            mid_value = matrix[mid//n][mid%n]
            if mid_value == target:
                return True
            elif mid_value < target:
                l = mid + 1
            else:
                r = mid -1
        return False


class Solution:
    def firstBadVersion(self, n: int) -> int:
        l = 1
        r = n

        while l <= r:
            mid = (l+r)//2
            if isBadVersion(mid):
                # it implies it is either the first or the consecutive
                # to check we need to move towards left
                # and check if the mid again is bad version
                # we need to do until the isBadVersion returns false
                r = mid-1

            else:
                # bad version exists to the right
                # shift left pointer right to mid
                l = mid+1
        return l # we need to move towards left until the l >r, and at the element, where isBadVersion fails, implies its right is the first bad version
        
#Now, let's consider the state of l and r when the loop exits:

#If the loop exits because l becomes greater than r, it means that r has been updated to a version that is smaller than l. This happens when isBadVersion(mid) returns True, indicating that mid is a bad version. In this case, r points to the last good version, and l points to the first bad version.
#On the other hand, if the loop exits because l becomes equal to r, it means that l and r converge to the same version. This happens when isBadVersion(mid) returns False, indicating that mid is a good version. In this case, l (and r) point to the first bad version.
#In both cases, when the loop exits, l points to the smallest version number that has not been eliminated as a good version. This is because:

#If l is pointing to a good version, the loop would have continued by updating l to mid+1.
#If l is pointing to a bad version, it means that all versions before l have been eliminated as good versions, and l is the first bad version.
    
# Tree Node

class TreeNode:
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None

# insert/delete node

# insert a node and return the root of BST
def insert(root,val):
    if not root:
        return TreeNode(val)
    if val > root.val:
        root.right = insert(root.right,val)
    elif val < root.val:
        root.left = insert(root.left, val)
    return root

def mininode(root):
    current = root
    while current and current.left:
        current = current.left
    return current

# remove a node from a tree
def remove(root,val):
    if not root:
        return None
    if val > root.val:
        root.right = remove(root,val) # remove and connect back to the original tree
    elif val < root.val:
        root.left = remove(root.left,val)

    else: # if found
        if not root.left:
            return root.right
        elif not root.right:
            return root.left
        else: # if has both the children
            minnode = mininode(root.right) # taking min node on the RHS helps to maintain the BST properties.
            root.val = minnode.val
            root.right = remove(root.right, minnode.val)
    return root  # to return back to the main recursive call and return the main root

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    
    def minvalue_node(self, root:Optional[TreeNode]) -> Optional[TreeNode]:
            current = root
            while current and current.left:
                current = current.left
            return current

    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        
        # if less than you work the same on the left node 
        # if greater than, you remove on the right node
        # if equal, i.e, we are at the node to be removed
        # if left does not exists, we remove the root by making the right node as the new root/ or return right node
        # similarly, if right doesnot exists, we return kleft node as the new root, as we are removing the current root
        # if it has both left and right, 
        # we look for the smallest on the right sub tree
        # and replace the root with the smallest to satisfy BST properties
        # and since we have two similar nodes (root, the same smallest at the other node)
        # and we need to remove the smallest value from that right sub tree 
        # by applying the same function on the same subtree

        if not root:
            return None
        if root.val < key:
            root.right = self.deleteNode(root.right,key)
        elif root.val > key:
            root.left = self.deleteNode(root.left,key)
        else: # if key found
            if not root.right:
                return root.left
            elif not root.left: 
                return root.right
            else: # if it has both children
                minnode = self.minvalue_node(root.right)
                root.val = minnode.val
                root.right = self.deleteNode(root.right,minnode.val)
        return root
        # one time to find minimum node log n
        # 2nd time to remove the minimum node, we need traverse again
        # it would be simple case, to check , if it has further right child or not 
        # (as min node wont have any further left nodes and we just assign right as the root)

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # inOrderTraversal - objective is to get the elements in ascending order
        # we visit the left subtree first, if it has left node, we visit it again
        # else, we move back to the root, then towards right, then upwards from there

        # if its an iterative way, how would we do ?
        # we need to use a data strcuture, where as we traverse, we store in it
        # and when we have to extract, the top one, which is the recent visited, comes first.
        # we can stire the passing nodes in the stack, although it does not cover all the nodes
        # it does cover all the main nodes whoich contribute to the length in the left sub tree
        """
        stack = [] # lest store nodes in stack, and it helps to traverse
        result = []
        current = root

        while current or stack: # initially stack will; be empty, we fill it and unfill it
            while current:
                stack.append(current)
                current = current.left

            # for the first time, stack must have all the left side
            # and it will stop once we reach the bottom
            # now we traverse back, one step/ level back and check if it has right node at this root
            current = stack.pop()
            result.append(current.val)

            current = current.right
            # now we do the same  for the right sub tree 
            # for leaf node right butre, much does not exist, we will just take that into stack and as the current becomes null
            # we pop that into result
            # for each left node, there will we one right node
            # and we collect them 
        return result
        """
        result = []
        def  traverse(node):
            
            if not  node:
                return # sreturn tback to the root node
            
            traverse(node.left) # traverse to the left until we mee t a null node
            # then it returns and appends to the result
            result.append(node.val)
            # traverse to the right once, then check it has left node
            # if it does not return to the root node, which now is the right node
            # append it and then check again
            traverse(node.right)
        
        traverse(root)

        return result

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# inorder: prioritizng left would be, left, parent, right child
def inorder(root):
    if not root:
        return
    inorder(root.left)
    print(root.val)
    inorder(root.right)

# preorder: prioritizing left would be: parent, left child, right child, 
def preorder(root):
    if not root:
        return
    print(root.val)
    preorder(root.left)
    preorder(root.right)

# postorder: prioritizing left would be: left, right, parent
def postorder(root):
    if not root:
        return
    postorder(root.left)
    postorder(root.right)
    print(root.val)

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:

        """
        if not preorder or not inorder:
            return None
        # root will be the first element in the preorder tree
        root = TreeNode(preorder[0])

        # in inorder, root divides elements into right and left subtree
        # hence we find its node in it
        mid = inorder.index(preorder[0])
        # since mid indicates the no. of elements in the left sub tree for a given node
        # we use this to slice both inorder and preorder arrays to allocate the left elements to the left sub tree
        # similarly right elements to the right sub tree
        # in preorder array, since first element is the root, we consider (mid+1) - 1 elements starting from index 1.
        root.left = self.buildTree(preorder[1:mid+1],inorder[0:mid])
        root.right = self.buildTree(preorder[mid+1:],inorder[mid+1:])

        return root
        
        """

        if not preorder or not inorder:
            return None
        
        # Create the root node with the first element of preorder
        root = TreeNode(preorder[0])
        stack = [root]
        inorder_index = 0
        
        for i in range(1, len(preorder)):
            # Get the top node from the stack
            node = stack[-1]
            
            # If the top node's value doesn't match the current inorder element,
            # create a new node as the left child of the top node
            if node.val != inorder[inorder_index]:
                node.left = TreeNode(preorder[i])
                stack.append(node.left)
            else:
                # If the top node's value matches the current inorder element,
                # pop nodes from the stack until we find a node that doesn't match
                # or the stack becomes empty
                while stack and stack[-1].val == inorder[inorder_index]:
                    node = stack.pop()
                    inorder_index += 1
                
                # Create a new node as the right child of the last popped node
                node.right = TreeNode(preorder[i])
                stack.append(node.right)
        
        return root

    # if node.val != inorder[inorder_index]:: If the top node's value doesn't match the current inorder element, 
    # it means we have encountered a left child. We create a new node with the current value from the preorder list 
    # and make it the left child of the top node. We append this new node to the stack.

    # while stack and stack[-1].val == inorder[inorder_index]:: If the top node's value matches the current inorder element, 
    # we keep popping nodes from the stack until we find a node that doesn't match or the stack becomes empty. 
    # We update inorder_index accordingly.

    #  Iterative solutions, like the stack-based approach, often have better performance compared to recursive solutions. 
    # Iterative code typically has fewer function calls, less overhead, and can be more cache-friendly. 
    # The stack-based approach leverages iteration to process the preorder and inorder traversals, 
    # which can lead to faster execution compared to the recursive approach.
    # It's important to note that the actual performance difference between the stack-based and 
    # recursive approaches may vary depending on factors such as the input size, 
    # the structure of the binary tree, and the specific implementation details. 
    # In some cases, the recursive approach may still perform well, especially for small input sizes or
    #  when the recursive calls are optimized by the compiler.


        
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

from collections import deque

class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        # level order traversal is nothing but breadth first search
        queue = deque()
        result = []
        if root:
            queue.append(root)

        while queue :
            level_size = len(queue)
            curr_level = []

            for _ in range(len(queue)):

                curr = queue.popleft()
                curr_level.append(curr.val)
                if curr.left:
                    queue.append(curr.left)

                if curr.right:
                    queue.append(curr.right)

            if curr_level:
                result.append(curr_level)
        return result


class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:


        """
        result = [[]]
        
        for num in nums:

            # for each element in the subset, you add an element
            # to current subsets, these subsets are the already sets in 
            # results

            new_subsets = [subset + [num] for subset in result]
            result.extend(new_subsets)

        return result
        """
        result = []

        # go through each element,
        # for each element combine it with all the rest of the elements
        # like for 1, it would [1], [1,2],[1,2,3]
        # what are we doing here, for each element, we hold a subset 
        # and add other elements to it incrementally using another pointer 
        # until we reach the end of the array 

        # for 2, the same index, will be 2 but we do it onl until 3
        # for 3, only [3] and we have already reached the end of the array
        
        def backtrack(start, subset):
            # consider start element and use subset for adding elemens to it
            # append it to result everytime we add something to subset

            # Append the current subset to the result
            # This is done for every subset, even an empty subset
            result.append(subset[:])

            for i in  range(start, len(nums)):
                # Include the current element in the subset
                subset.append(nums[i])
                # we need to add this to result and go to the next element
                # hence we increment the pointer and call the fundstion again

                # Append the current subset to the result
                # This is done for every subset, even an empty subset

                # Recursively call backtrack with the next index (i+1) and the updated subset
                # This explores all subsets that include the current element
                backtrack(i+1, subset)
                # once this backtract we remove that element from the subset
                # for the subset to start from the scratch
                # after [1],[1,2],[1,2,3] we pop out 3, then 2, then i reaches 2 
                # and should be appended to  [1] to get [1,3]

                # Backtrack by removing the last added element from the subset
                # This is done to explore subsets that exclude the current element
                subset.pop()
                # After popping, the subset is restored to its state before the current element was added

        # Start the backtracking process with an empty subset and starting index 0
        backtrack(0,[])
        return result
O(2)
    
"""
    
backtrack(0, [])
  append [] to result
  i = 0
    append 1 to subset
    backtrack(1, [1])
      append [1] to result
      i = 1
        append 2 to subset
        backtrack(2, [1, 2])
          append [1, 2] to result
          i = 2
            append 3 to subset
            backtrack(3, [1, 2, 3])
              append [1, 2, 3] to result
              return
            pop 3 from subset
          return
        pop 2 from subset
      i = 2
        append 3 to subset
        backtrack(3, [1, 3])
          append [1, 3] to result
          return
        pop 3 from subset
      return
    pop 1 from subset
  i = 1
    append 2 to subset
    backtrack(2, [2])
      append [2] to result
      i = 2
        append 3 to subset
        backtrack(3, [2, 3])
          append [2, 3] to result
          return
        pop 3 from subset
      return
    pop 2 from subset
  i = 2
    append 3 to subset
    backtrack(3, [3])
      append [3] to result
      return
    pop 3 from subset
  return

result = [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]
"""

# creating min/ max heap
# for element at i, its children are at indices 2*i+1 and 2*i+2
# for an element at i, its parents are at (i-1)//
class Heap:
    def __init__(self):
        self.heap = []

    def push(self,val):

        self.heap.append(val)
        i = len(self.heap) - 1

        # we add element at the end and as long as it is less than the parent, we swap it
        while i >= 0 and self.heap[i] < self.heap[(i-1)//2]:
            tmp = self.heap[i]
            self.heap[i] = self.heap[(i-1)//2]
            self.heap[(i-1)//2] = tmp
            #move to the parent's index and check with the parent's parent
            i = (i-1)//2

    def pop(self):

        # remove the root of the array
        # and we can replace with the child
        # but, it creates a void if the nodes are not perfectly the same and balanced on both sides
        # hence better way is to replace with the one at the bottom.
        # and then push it down by checking with each of the chidren

        if len(self.heap) == 1:
            return self.heap.pop()
        root = self.heap[0]

        # replace the root with the last element in the heap
        self.heap[0] = self.heap.pop()

        size = len(self.heap)
        i = 0
        while True:
            # we want to keep the smallest at the top
            smallest = i
            left = 2*i+1
            right = 2*i +2

            # if left element exists (i.e., left is within the size) and is less than the current root
            if left < size  and self.heap[left] < self.heap[i]:
                smallest = left
            if right < size and self.heap[right] < self.heap[i]:
                smallest = right

            # if the current element is not the smallest, swap it as long as current element is the smallest
            if smallest != i:
                self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
                i = smallest

            else:
                break

        return root
    
class MaxHeap:
    def __init__(self):
        self.heap = []

    def push(self, val):
        self.heap.append(val)
        i = len(self.heap)
        while i >0 and self.heap[i] > self.heap[(i-1)//2] :
            tmp = self.heap[i]
            self.heap[i] = self.heap[(i-1)//2]
            self.heap[(i-1)//2] = tmp
            i = (i-1)//2
    

# for a min heap of size k, kth largest is the root. isnt it
# so, just by maintaining a min heap of size k , we can always return kth largest
# everytime we see a new element, if its smallest than the current root, it cannot be among the kth largest, hence we discard it.
# however, if its largest, it should be among, and we push it to the heap and return the root

# maintain a min heap of k size and return the root
class kthlargest:



    def __init__(self, k:int, nums:List[int]) :
        self.heap = []
        self.k = k 
        for num in nums:
            self.add(num)

    def add(self, val):

        self.push(self.heap,val)

        if len(self.heap) > self.k:
            self.pop(self.heap)
        return self.heap[0]
    
class Node :
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCache:

    """
    # we can do it as an array of key,value tuples
    # for putting it, we first check, if it already exists, 
    # 1. objective for putting is to place it at the end to mark it as most recently used
    # 2. hence, if it already exists, we remove it, else we append it at the end
    # 3. if it doesnot exist and we are adding a new element, 
    # we check, if the length is already exceeded and we need to pop the least used one
    # that is the starting one and the append at the last
    # for getting it, if its already there, we return it, else we return -1
    # however, as we are getting it again, this becomes the most recent one
    # hence, we need to remove it and append it at the end

    # However, with dynamic arrays it will be O(n) for insertion, and deletion
    # hence its best to chose another data structure.
    def __init__(self, capacity: int):
        self.cache = []
        self.capacity = capacity

    def get(self, key: int) -> int:
        for item in self.cache:
            if item[0] == key:
                self.cache.remove(item)
                self.cache.append(item)
                return item[1]
        return -1
        

    def put(self, key: int, value: int) -> None:
        for item in self.cache:
            if item[0] == key:
                self.cache.remove(item)
                break
        if len(self.cache) >= self.capacity:
            self.cache.pop(0)

        self.cache.append((key,value))

    """

    def __init__(self, capacity: int):
        self.capacity  = capacity
        self.cache = {}
        self.head = Node(0,0)
        self.tail = Node(0,0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self,node):
        prev, next = node.prev, node.next
        prev.next, next.prev = next, prev

    def _add(self,node):
        prev, next = self.tail.prev, self.tail
        prev.next = node
        next.prev = node
        node.prev = prev
        node.next = next

    def get(self,key):
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add(node)
            return node.val
        else:
            return -1

    def put(self,key, val):
        if key in self.cache:
            self._remove(self.cache[key])
        node = Node(key, val)
        self._add(node)
        self.cache[key] = node
        if len(self.cache) > self.capacity:
            lru = self.head.next
            self._remove(lru)
            del self.cache[lru.key]

    

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        
        """
        anagrams = {}
        # sorted strings have similar character sequence
        # and hence can be anagrams of each other
        for s in strs:
            key = "".join(sorted(s))
            if key not in anagrams:
                anagrams[key] = []

            anagrams[key].append(s)

        return list(anagrams.values())

        # but this takes n * klogk time, n for strings and k for length of each string
        # storage is of O(n*k), n for length and k for strings
        # we can further go ahead in reducing time, by considering count as key instead of sorted string
        """
        
        
        anagrams = {}

        for s in strs:

            letterCounts = [0]*26
            for l in s:
                idx = ord(l) - ord("a")
                letterCounts[idx] += 1
            
            # for hashing we should have immutable object as key, hence lets turn that into a tuple

            t = tuple(letterCounts)

            if t not in anagrams:
                anagrams[t] = []
            anagrams[t].append(s)

        return list(anagrams.values())
"""
Complexity Analysis:
Time Complexity: The time complexity of this function is O(n×k), where 
n is the number of strings and 
k is the average length of the strings. This is because for each string, you're iterating over its characters once to count frequencies.
Space Complexity: The space complexity is 

O(n×k) in the worst case, considering the storage for the dictionary and the lists of anagrams. Although the key (tuple) space is constant (26-length tuple), the list storage will vary with the number and size of the strings.
Advantages Over Sorting:
Speed for Character Counting: Counting characters might be more efficient than sorting when strings are very long but character diversity is low (e.g., many characters are repeated).
Constant Key Size: The tuple used as a key has a constant size (26) regardless of the string length, whereas sorted strings as keys would vary based on string content and length.

"""


### HASHSET

# A hashset typically should be used for quick checking if something exists in a set, add if new, remove if we have to
# lets start implementing with arrays

class HashSet:
    def __init__(self):
        self.set = []

    def contains(self,key):
        if key in self.set:
            return True
        else: 
            return False
    def add(self,key):
        if not self.contains(key): # takes O(n) as it has to check with element
            self.set.append(key)

    def remove(self,key):
        if key in self.set:
            self.set.remove(key) # every operation is O(n)


# it can be optimized by optimizing search operation like having ordered elements and doing binary search in the conjunction
# like when you place keys we place them in a sorted order
# it reduces the search time for finding key to O(logn) however, to maintain the order, we still need to do O(n) operations everytime, we insert a new element
# contains : O(logn) for search, add, remove : also O(logn) for search, but still can be O(n) if we have to shift the elements for maintaing order
 
class HashSet():
    def __init__(self):
        self.set = []

    def contains (self, key):
        index = self.binary_search(key)
        if index < len(self.set):
            if key == self.set[index]:
                return True
        return False
    def binary_search(self,key):
        # has to return the index of the key if it exists
        l, r = 0, len(self.set) - 1
        while l<=r:
            mid = (l+r)//2
            if self.set[mid] == key:
                return mid
            elif self.set[mid] > key :
                r = mid-1
            else:
                l = mid+1
        # if not found till now, it implies we have reached a point where r==l, 
        # and we return it as the place where the new element is to be added
        return l # if not found return the last smallest index greater than the last filled item
    def add(self, key:int) -> None:
        index = self.binary_search(key)
        # add at the end, only not already in the set
        # and adding at the end implies, all other elements are less than the key and low has reached the length of the array

        # also we need to check there are no duplicate elements, 
        # [10,20,50] we need to add 30, so l returns 2, and we need to check 50! = 30
        if index == len(self.set) or key != self.set[index] :
            self.set.insert(index,key )

    def remove(self,key):
        index = self.binary_search(key)
        if index < len(self.set) and self.set[index] == key:
            self.set.pop(index)

# We can improve on this further by oprimizing insertion and removal times as well.
# currently for binary search, we have O(n) for a sorted array, if we have to place/remove it in between
# along with binary search type complexity O(logn) for comtains, we need something which offers O(logn) complexity for add/remove
# BSTs come to the rescue and this removes the need for constant resizing in case of arrays, when new elements are added.

# also, this is best suitable for BST, as they do not allow duplicates in their structure

class TreeNode():
    def __init__(self,key):
        self.left = None
        self.right = None
        self.val = key # value of the node

class HashSet():
    def  __inti__(self):
        self.root = None

    def add(self, key):
        if not self.contains(key):
            self.root = self._insert(self.root, key)
    def remove(self,key):
        self.root = self._delete(self.root, key)
    
    def _delete(self,node, key):
        if not node:
            return node
        if key < node.val:
            node.left = self._delete(node.left, key)
        elif key > node.val:
            node.right= self._delete(node.right, key)
        else:
            # key == node.val found

            # we need to remove this node
            # implies, we need to connect its childrent to the node's parents
            if not node.left:
                return node.right
            elif not node.right:
                return node.left
            else:
                # if both the children exist
                # we find the minimum node in the right side or greater than side. 
                # and replace current node with this min node to maintain the properties of BST 
                # left side will be less than this point and right side witll be greater than this point
                min_node = self._find_min_node(node.right)
                node.val = min_node.val
                # now we need to remove the redundant node
                # at last it become none or we return left or right node
                node.right = self._delete(node.right, min_node.val)


        return node # we need to return the unchanged node, to maintain the link. 
            # we change its children and return the node back, like in case of recursive calls within the function 
            # we either change the left or right sub strcutures, then the within recursive call ends by adding new node and the reaches
            # the end line where we return the main node

    def _find_min_node(self, node):
        current = node
        while current.left is not None:
            current = current.left

        return current

    def contains(self,key):
        # search if a key exists in the tree
        return self._search(self.root,key)
    
    def _search(self, node, key):
        if not node:
            return False
        elif node.val == key:
            return True
        elif node.val < key:
            # search right side
            return self._search(node.right, key)
        else:
            # search left side
            return self._search(node.left, key)
        
    def _insert(self, node, key):
        # recursively insert the new key and return the new root

        if not node: # if no node, we create a new Node
            return TreeNode(key)
        elif node.val < key:
            node.right =  self._insert(node.right, key)
        else:
            node.left = self._insert(node.left, key)
        return node #  we need to return node for the link maintenance. During each recursion call, if we return to the node, we maintain r/nship between parent and child

# if space is not of concern, we can just initialize a large array

class HashSet():
    def __init__(self):
        self.size = 1000
        self.table = [False]*self.size

    def add (self, key):
        if key < self.size:
            self.table[key] = True
    def remove(self, key):
        if key < self.size:
            self.table[key] = False

    def contains(self, key):
        return self.table[key] and key < self.table[key]
    
# extremely fast operations O(1), but requires very high space for allocation of arrays
    

# linked Bucket Lists - each bucket is a linked list and avoids collision by chaining together elememnts that hash to the same index
# ideally, if the hash function distributes elements uniformly, O(1) is the time complexity for all the operations
# however, if all the elements hash to the same bucket, performance degrades to O(n)

# if tree is balanced BST offers better average case performance, with operations chieving O(logn), but if the keys are already sorted it goes 
# down to O(n)
# BSTs are preferred when order needs to be preserved or guaranteed
# if hash function can guarantee to distribute elements evenly, Linked Lists are preferred.
# generally, for quick lookup, insertion, deletion, without regard to space, then Linked List are prefeeered
# however, if we need ordered data traversal, or where the datasets is changing dynamically, BSTs are preferred.
# useful, where inherent sorting of elements can be leveraged for quick searches, range queries etc.

### Linked Bucket Lists
# we maintain an array of buckets
# each bucket corresponds to an index from the has function
# if more than one elements gives the same hash value, then we chain it to the existing node in the bucket.

class Node():
    def __init__ (self,key, next = None):
        self.val = key
        self.next = next

class HashSet():
    def __init__(self):
        self.capacity = 1000  # Adjust capacity based on expected number of operations
        self.buckets = [None]*self.capacity

    def hash(self, key):
        return key % self.capacity
        
    def contains(self, key):
        index = self.hash(key)
        current = self.buckets[index]
        while current:
            if current.val == key:
                return True
            current = current.next

        return False
    def add(self, key):
        index = self.hash(key)
        if not self.contains(key):
            node = Node(key,self.buckets[index])
            self.buckets[index] = node

    def remove(self, key):
        index = self.hash(key)
        current = self.buckets[index]
        prev = None
        while current:
            if current.val == key:
                if prev:
                    prev.next = current.next
                else:
                    self.buckets[index] = current.next

            prev = current
            current = current.next

"""

 Buckets:
[0] -> None
[1] -> None
[2] -> None
[3] -> None
[4] -> None

Adding Key 2    
Hash index for 2 might be calculated as 2 % 5 = 2.
Since bucket [2] is empty, a new node with key 2 is created and becomes the head of the list at bucket [2].
[0] -> None
[1] -> None
[2] -> Node(2) -> None
[3] -> None
[4] -> None

Adding Key 7
Hash index for 7 is 7 % 5 = 2.
This key also maps to bucket [2]. A new node with key 7 is added at the head, pushing the previous head (Node(2)) backward.
State After Adding Key 7:

[0] -> None
[1] -> None
[2] -> Node(7) -> Node(2) -> None
[3] -> None
[4] -> None

Visualizing the remove Operation
Let’s continue with the previous scenario and now remove keys from the HashSet.

Removing Key 7
Hash index for 7 is still 2.
Key 7 is found at the head of bucket [2]. The head is then updated to the next node in the list, which is Node(2).
State After Removing Key 7:

[0] -> None
[1] -> None
[2] -> Node(2) -> None
[3] -> None
[4] -> None
"""

class TreeNode():
    def __init__(self, key, value, next = None):
        self.key = key
        self.val = value
        self.next = next

class MyHashMap:
    def __init__(self):
        self.capacity = 10000
        self.buckets = [None]*self.capacity

    def hash(self, key):
        return key % self.capacity
    
    def contains(self, key):
        idx = self.hash(key)
        current = self.buckets[idx]

        while current:

            if current.key == key:
                return True
            current = current.next
        return False
    
    def get(self, key):

        idx = self.hash(key)
        current = self.buckets[idx]
        # the current is always set to head
        # so need not worry about missing previous
        while current:
            if current.key == key:
                return current.val
            
            current = current.next

        return -1
    
    def put(self, key, value):
        idx = self.hash(key)
        current = self.buckets[idx]
        while current:
            if current.key == key:
                current.val = value
                return
            current = current.next
        # if not self.contains(key):
        node = TreeNode(key,value, self.buckets[idx])
        self.buckets[idx] = node

    def remove(self, key):
        idx = self.hash(key)
        current = self.buckets[idx]
        prev = None

        while current:
            if current.key == key:
                if prev:
                    prev.next = current.next

                else:
                    self.buckets[idx] = current.next
                return

            prev = current
            current = current.next


# Matrices 
# Count Unique Paths

"""
Fully Connected Grid or Graph: In a worst-case scenario for DFS, 
such as a large grid where each cell is connected and 
you could potentially move from one corner of the grid to the opposite corner (or cover all cells in some pathfinding scenarios), 
the recursion might need to go as deep as the total number of cells in the grid. 
If every cell is visited before the recursion starts to unwind (return), the recursion stack could have one frame for each cell.
Space Complexity Formulation: Hence, the space complexity is described as O(M×N) for a grid. 
This means that in the worst case, every cell in the grid could be part of the recursion stack at some point if you follow a path covering all cells before hitting a base case.
"""

class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:

        """

        if not obstacleGrid :
            return 0

        m, n = len(obstacleGrid), len(obstacleGrid[0])

        def countUniquePaths(r,c):



            # only right and downward
            # if r >= len(obstacleGrid) or r < 0 or c >= len(obstacleGrid[0]) or c < 0 :
            if r >= m or c >= n or obstacleGrid[r][c] ==1 :
                return 0

            if r == m-1 and c == n-1 :
                return 1


            return countUniquePaths(r+1,c) + countUniquePaths(r,c+1)

        return countUniquePaths(0,0)

        """
        # time limit exceeding for 2^m*n operations O(2^m*n) and space complexity O(m*n)
        # we perhaps calculate the same paths for each node again and again
        # and we can use the previous calculations to minimize the calculations on a give node

        if not obstacleGrid or obstacleGrid[0][0] == 1:
            return 0

        m, n = len(obstacleGrid), len(obstacleGrid[0])
        memo = {}

        def dfs(r,c):

            if (r,c) in memo:
                return memo[(r,c)]

            if r >= m or c >=n or obstacleGrid[r][c] == 1 :
                return 0

            if r == m-1 and c == n-1 :
                return 1

            memo[(r,c)] = dfs(r+1,c) + dfs(r,c+1)

            return memo[(r,c)]

        return dfs(0,0)


# No. of islands
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:

        count = 0
        if not grid:
            return 0

        def dfs(r,c):

            if r <0 or r >= len(grid) or c >= len(grid[0]) or c <0 or grid[r][c] == "0":
                return 


            # mark current visited node as 0
            grid[r][c] = "0"

            dfs(r+1,c)
            dfs(r-1,c)
            dfs(r,c+1)
            dfs(r,c-1)

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "1":
                    dfs(i,j)
                    count +=1

        return count



class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        
        if not grid:
            return 0


        def dfs(r,c):

            if r<0 or r >= len(grid) or c<0 or c>= len(grid[0]) or grid[r][c] == 0:
                return 0

            grid[r][c] = 0

            area = 1

            area += dfs(r+1,c)
            area += dfs(r-1,c)
            area += dfs(r,c+1)
            area += dfs(r,c-1)

            return area

        max_area = 0

        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == 1:
                    max_area = max(max_area, dfs(r,c))

        return max_area

## BFS

"""
The space complexity in the worst case can be up to O(N 2), N^2 represents the total number of cells in the n×n grid. 
This would be the case if almost every cell needs to be processed and stored in the queue at some point during the BFS execution, 
such as in a densely packed grid where most cells are '0' and every cell is explored.

In real-world applications, this space complexity implies that BFS can be memory-intensive, particularly for large grids. 
The need to potentially hold a significant portion of the grid in memory (in the queue) can impact performance and feasibility, 
depending on the system's available memory.

The time complexity can be expressed as O(8×N^2) because each of the N^2
nodes could potentially cause up to 8 operations (checking each neighbor). Simplified, this becomes 
O(N 2), as constant factors are generally omitted in Big O notation.

In BFS for general graphs, every vertex is visited once, and every edge is traversed once. This leads to the 
O(V+E) complexity. The total operations are based on the sum of the number of vertices (as each is visited once) 
and the number of edges (as each is considered once to move from one vertex to another).


O(n^2) IN GRIDS: The BFS in a grid situation might need to explore all cells (nodes) in the worst case (where 
N=n^2 for an n×n grid).
"""

from collections import deque
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        
        m, n = len(grid), len(grid[0])
        queue = deque()
        fresh_count = 0

        #count the number of rotten oranges and push them to a que
        # also lets add a time dimension to these states in the que
        # that is if there is a rotten orange at [1,1] at the beginning
        # lets add time elapsed to the same tuple as  (1,1,0)
        # and then we can use this time elapsed to update the time 
        # while propagating the rot

        for r in range(m):
            for c in range(n):
                if grid[r][c] == 2 :
                    queue.append((r,c,0))
                elif grid[r][c] == 1:
                    fresh_count += 1

        if fresh_count == 0:
            return 0

        time_elapsed = 0
        dir = [(0,1),(1,0),(0,-1),(-1,0)]

        while queue:
            r,c, minute = queue.popleft()
            for dr, dc in dir:
                nr, nc = dr+r, dc + c
                if 0<= nr< m and 0 <= nc < n and grid[nr][nc] == 1:
                    # make it rot
                    grid[nr][nc] = 2
                    # append it to the queue later to propagate from that node as well
                    queue.append((nr,nc,minute+1))
                    # decrement fresh orange count
                    fresh_count -=1
                    time_elapsed = minute + 1

        if fresh_count:
            return -1
        else:
            return time_elapsed
                   



"""
space I guess is O(m*n) for holding the queue if all the oranges are srotten. and time complexity is also O(2* m*n), 
as we visit each node at least once ? once for finding rotten ones, once for finding non rotten ones

"""


"""
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

"""

from collections import deque
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        
        # need to find the shortest distance to 0
        # for each cell

        if mat is None:
            return mat

        # lets initialize returns
        m , n = len(mat), len(mat[0])

        #result = [ [0 for _ in range(n)] for _ in range(m)]
        result =[[float('inf') if mat[i][j] == 1 else 0 for j in range(n)] for i in range(m)]

        # from the example, for each cell, we can go in 4 different directions
        dir = [(0,1),(1,0),(-1,0),(0,-1)]

        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    result[i][j] = 0
                else:
                    # for each node, we search across the neighbours
                    # if its one
                    # we search for its neighborss
                    # they could be 0 or 1
                    # if any of the neighbors is 0
                    # we make the distance corresponding to it 1
                    # else we make the distance 2 
                    # and we add them to the queue until as they have not been processed yet
                    # we would be stopping only when we find any of the neighbors to be 0
                    # else, we add them to the que
                    # so, the queues in the next layer, would be ones
                    # which did not find any 0's in their neighbors

                    # have to aware that in grid/graph problems, layers/ ques 
                    # are supposed to desrcibe elements which have common properties
                    # like all 0's belong to the 1st layer
                    # all the 1's with 0's a immediate neighbors are supposed to be 2nd layer
                    # and so on

                    # however, if we search from the 1's perspective, we might 
                    # be traversing the same elements again and again.
                    # hence better to work from 0's towards 1's.

                    queue = deque([(i,j,0)])
                    visited = set((i,j))
                    found = False 
                    while queue and not found:
                        r,c, dist = queue.popleft()
                        for dr, dc in dir:
                            nr, nc = r+dr, c +dc
                            if 0<= nr< m and 0<= nc < n and (nr,nc) not in visited:
                                if mat[nr][nc] == 0:
                                    result[i][j] = dist + 1
                                    found = True
                                    break
                                else:
                                    queue.append((nr,nc,dist+1))
                                    visited.add((nr,nc))

        return result

        """
        Time Complexity
        BFS from Each 1:
        For each cell that is a 1, you potentially perform a BFS to find the nearest 0. 
        During this BFS, in the worst case, you might end up exploring many or all other cells in the matrix.
        Suppose there are k cells that are 1 in an m x n matrix. 
        For each 1, you may, in the worst case, explore the entire matrix if 0s are sparse or located far from these 1s. This results in a complexity that could approach O(k * m * n).
        
        Worst-Case Scenario:
        In the worst-case scenario where the matrix is dense with 1s except for a few 0s, nearly every cell initiates a BFS. If the number of 1s (k) approaches m*n, then the time complexity in the absolute worst case can be as high as O((m * n) * (m * n)) = O(m^2 * n^2). This quadratic complexity occurs because each BFS might need to explore a significant portion of the matrix.
        
        Space Complexity
        Queue and Visited Set:
        The maximum size of the queue can, in the worst case, 
        hold a significant fraction of the matrix if many paths are simultaneously exploring multiple directions. The visited set will hold at least as many elements as are in the queue at any time.
        In the worst case, where BFS needs to explore large parts of the matrix from many starting points, the space complexity can approach O(m * n) due to the queue and the visited set together storing a number of elements proportional to the size of the matrix.
        
        Result Matrix:
        You also maintain a result matrix of size m x n, which stores the distances. However, this is not additional space in terms of complexity analysis since it's required for the output and matches the input size.
        
        """


    

        # instead we can start from 0's and move towards 1s like rotten oranges.
        # lets start from the layer which has all 0's and it is the base layer as it 0 distance from 0's.
        # now we look into the neighbors of the elements in this layer
        # if they are zeros, its fine, we do not add them to the que, else we add them to the que and 
        # update their distance to 0+1 = 1 as they are only 1 step away from the 0's layers
        # once we are done with the 1st layer, we move to the next layer, 
        # then look into their neighbors, if they are 0, great.. 
        # Hmm, nope this condition doesnot consider the simultaneous update of all the neighbors of all the zeros
        # we need a better condition to check, we can check their current result from 0's.
        # if they are infinity as we initiated earlier, it implies they are untouched
        # then we add 1 unit distance to the parent cell's distance to 0th layer and allocate the new distance to the cell.
        # else, if they are already tocuhed, implies they are 1 unit away from the zeroth layer, we do nothing
        # the better condition is current distance of neigh bour from 0 if greater than the parent cells distance + 1
        # if greater, we update the neighbor cell's distance with parents distance +1, as it is the minimum possible distance
        # if we check pure distance without 1, we are forgetting that we are checking the neighbor of the cells and 
        # that we are checking if a cell and its neighbors both are at the same distance from the nearest possible 0.
        
        
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:        
        if not mat:
            return -1

        m,n = len(mat),len(mat[0])

        result = [[0 if mat[i][j]==0 else float("inf") for j in range(n) ] for i in range(m)]
        
        queue = deque()
        # know where the 0's are currently
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    queue.append((i,j))

        

        dir = [(0,1), (1,0),(-1,0),(0,-1)]
        while queue:
            r,c = queue.popleft()
            for dr,dc in dir:
                nr, nc = dr + r , dc+ c
                if 0<= nr < m and 0<=nc < n and result[nr][nc] > result[r][c] + 1 :
                    result[nr][nc] = result[r][c] + 1

                    queue.append((nr, nc))

        return result 

        # we visit each node once to get 0's and 1's, hence its O(m*n)
        # then while doing BFS, we consider BFS and its neighbors again, 
        # and we update the distance of neighbors, only if further shortest distance can be found.
        # this operation can again go not greater than O(m*n). 
        # coz once we found the nearest distance from a 0, we most probably wont be updating it again
        # hence less additions to the que as we move forward.

        # space complexity : O(m*n), 1 for queue, 1 for result set.



from collections import defaultdict

# we use default dict for preparing adjacency list, useful when creating new adjancency list for a graph
# like 0 is a pre requisite for courses both 1 and 2
# like it would be helpful if we dont know keys and encounter them dynamically
# but here we know the nodes.

# prereq = {0:[1,2]}
#  and we traverse from 0 and check for the prerequisites of 1 and 2
# if any of them have 0 as pre requisite then its done, we found a cycle
# and we return False as the completion is impossible

# only thing is that we need to know that we have already tracked 1 and 2.
# how do we do this ??
# lets say we are at a course 0, and we mark it as visiting
# now we move to the next layers, that is the next level courses 1  and 2
# and we mark them again as visiting, as we are dealing with them for the first time.
# and as we move to the courses, for which each of the courses 1 and 2 is prerequisites
# and if we find 0 there again, and we can find it by looking at the flag of 0, if its marked as visiting
# it implies, we have to study 0 again while using it as a prerequisite for the 2nd layer courses
# hence cycle found, else all good.

# question says that there are n-1 courses labeled from 0 to n-1
# we can track all the n courses in visit window
# if 0, not yet visited, 1 visiting, 2 visiting done

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:

        #next_courses = defaultdict(list)

        next_courses= {i: [] for i in range(numCourses)}

        # lets construct an adjancency list, where we have current courses, which would lead to next courses
        # like if  0 is a prereq for 1,2, then 0, leads to both 1 and 2 {0:[1,2]}

        for dest, src in prerequisites:
            if src >= numCourses or  dest >= numCourses:
                return False
            next_courses[src].append(dest)


        visited = [0]*numCourses

        def dfs(course):

            if visited[course] == 1 : # if visiting again
                return False # entered cycle

            if visited[course] == 2:
                return True # completely processed a course, without encountering any cycle


            # visiting course for the first time
            # set to visiting
            visited[course] = 1

            # recurse on all the course, that depends on this course
            for next_course in next_courses[course]: # we should not find a cycle
                if not dfs(next_course):             # it implies every node that we are seeing should not already be in a visited mode
                    return False            # here we are trying to work on all the nodes

            visited[course] = 2 # if it broke no where till now, mark it as fully processed.

            return True


        # start dfs on each course if not visited already
        for i in range(numCourses):
            if visited[i] == 0:
                if not dfs(i): # you cannot complete dfs without a cycle
                    return False

        return True

        # prerequisites represented by [1,0] represents a directed edge from 0 to 1, 
        # so, to process a node for course 1, we need to process the node and also the prerequisites
        # constructing adjacency lists, requires all the edges this takes O(E)
        # doing DFS involves traversing each node and also the each edge in the worst case. hence O(E+V)

        # space complexity: adjacency list stores all the edges and vertices .O(V+E)
        # also a bit for visited status, for all the nodes O(V) and recusrion tack depth can also go up to O(V)
        # in the case of no cycles, with linear dependency upto V.
        # so space complexity is O(V+E)

class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums)

        dp = [0]*len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1]) # you either keep the amount you have till now, 
        # or to take whats available in the second house

        for i in range(2, len(nums)):
            # rob only if ot yields value of robbing and adding it to 1 unit from previous house is greater than than previous valu
            dp[i] = max(dp[i-1], nums[i]+dp[i-2])

        return dp[-1]

        # space complexity is O(n) to store amount stored status.
        # time complexity is O(n) as we just go through each node once.

        

class Solution:
    def climbStairs(self, n: int) -> int:
        # fibonacci sums

        if n==1:
            return n
        if n==2:
            return 2
        
        dp = [0]*(n+1)
        dp[0] = 1
        dp[1] = 1

        for i in range(2,n+1):
            dp[i] = dp[i-1] + dp[i-2]

        return dp[n]

class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        
        # longest common subsequence
        #
        # we 

        """
        def LCS(i,j):
            if i< 0 or j < 0:
                return 0

            if text1[i] == text2[j]:
                return 1+ LCS(i-1, j-1)
            
            if text1[i] != text2[j]:
                return max(LCS(i,j-1), LCS(i-1,j))

        return LCS(len(text1)-1, len(text2)-1)
        """

        # however, for each cell, we are doing two function calls
        # that is for equal to or not equal to
        # and also, we may be doing the same calculations again in
        # LCS(i, j-1) and LCS(i-1, j) e.g. LCS(i-1,j-1)
        # this is brute force and doesnot involve any optimization
        # AND the time complexity is 2^O(m+n), where m and n are the length of the strings
        # at any point we decrement either i, or j or both in diagonal way
        # lets say if they dont match at all, we might have a recursion call, who length
        # is m+n , as decrement i,j from m-1 to 0
        # thus it is O(m+n)
        # space complexity comes from recursion stack and each call uses some
        # stack space, in the worst case, we might have to store the entire lengths of both the strings
        # as we traverse them together and hence space complexity = O(m+n)
        # maximum depth of the recursion = m+n

        # hence it takes more time
        # since we know that LCS(i-1, j-1) is being calculated for both
        # LCS(i, j-1) and LCS(i-1, j), we can store these calculations once and reduce the redundancy

        # may be dictionary and call stack is causing too much overhead
        
        """
        memo = {}

        def LCS(i,j):
            if i< 0 or j<0 :
                return 0

            if (i,j) in memo:
                return memo[(i,j)]
            
            if text1[i] == text2[j] :
                memo[(i,j)] = 1 + LCS(i-1, j-1)
                # return memo[(i,j)]

            else: #text1[i] != text2[j]:
                memo[(i,j)] = max(LCS(i-1,j) , LCS(i,j-1))
                # return memo[(i,j)]

            return memo[(i,j)]

        return LCS(len(text1)-1, len(text2)-1)

        """
        ###

        m, n = len(text1), len(text2)
        dp = [[0]* (n+1)  for _ in range(m+1)]

        for i in range(1,m+1):
            for j in range(1, n+1):
                if text1[i-1] == text2[j-1] :
                    # if matches, we take the dp value including both the characters till the last time
                    dp[i][j] = 1 + dp[i-1][j-1] #move to the next element for matching
                else:
                    # if doesnot match, we consider dp value, which ever gives the maximum value, by excluding the not matching character.
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]



class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n>0:
            if n & 1 == 1:
                count += 1  # if anding with 1 works, it implies they both have 1 in the units place
            n = n >> 1
            # this increments binary operations to the right
        return count

        """
        The purpose of this operation in the context of counting 1s in the binary representation 
        is to progressively reduce n by removing the examined least significant bit (LSB) 
        after it's been accounted for:

        Progress Through Bits: Each bit shift exposes a new LSB to be checked (if n & 1 == 1). 
        This way, each bit of the original n is eventually checked as it 
        becomes the LSB through successive right shifts.


        The time complexity of this function is O(k), where k is the number of bits in the number n. 
        Since the function iterates through each bit of the integer once, 
        the complexity directly depends on the bit-length of the number. 
        For a standard 32-bit integer, this would effectively be O(32), 
        which is considered O(1) under the assumption of a constant bit length. 
        For arbitrary-length integers, the complexity is O(log n), where n is the value of the integer 
        (as the number of bits required to represent n is proportional to the logarithm of n).

        Space Complexity:
        The space complexity is O(1) because the amount of space used does not depend on the size of the input 
        but only on a fixed number of local variables (count and n). 
        """



class Solution:
    def countBits(self, n: int) -> List[int]:

        """
        ans = [0]*(n+1) # n+1 to include no. of 1's in binary representation of 0 as well
        for i in range(n+1):
            count = 0
            x = i
            while x > 0:
                if x & 1 == 1:
                    count += 1
                x = x >> 1

            ans[i] = count

        return ans

        """
        # time complexity: each number has log n bits on average and since we count bits for every 
        # i upto n, it will be O(nlogn)
        # space complexity = O(n) as we are storing the counts of 1 for each number

        # however, we can use the previous calculated values
        # The number of 1s for any integer i can be quickly computed by 
        # using the results from integer i >> 1 (i.e., i/2) (in binary representation, 
        # shifting  by 1 bit implies dividng by 2)and 
        # adding 1 if the least significant bit of i is 1.

        ans = [0]* (n+1)
        for i in range(1, n+1):
            ans[i] = ans[i >> 1] + (i & 1)
        return ans

        """
        Time Complexity:
        O(n): In the dynamic programming approach where each number i's count of 1s is determined by ans[i] = ans[i >> 1] + (i & 1), 
        each number from 0 to n is processed exactly once. 
        The computation for each number involves a constant-time bit-shift operation and a constant-time bitwise AND operation, 
        plus a constant-time addition and array indexing operation. 
        Hence, the overall time complexity is linear with respect to n.

        Space Complexity:
        O(n): The space complexity is primarily dictated by the need to store the count of 1s for each number from 0 to n.
        This requires an array of length n+1, so the space complexity is linear with respect to n.
        """




class Solution:
    def reverseBits(self, n: int) -> int:
        reversed = 0
        for _ in range(32):
            # shift reversed by 1 bit to make space for the extracte lsb from n
            reversed = reversed <<  1
            # extract the LSB in n
            LSB = n & 1

            # add it to the right most portion
            reversed = reversed | LSB

            # then again shift it towards left at every iteration for 32 times

            n = n >> 1

        return reversed

        # time complexity = O(1) we act on each bit incrementally and its only 32 times constant
        # space complexity is also constant O(1)

class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:

        """
        newNums = nums + nums[:-1]

        currSum = maxSum = newNums[0]

        arr_i = 0
        for i in range(1, len(newNums)):

            if currSum < 0:
                currSum = newNums[i]
                arr_i = i
            else:
                currSum += newNums[i]

            if i - arr_i>= len(nums)-1:
                
                currSum -= newNums[arr_i] 
                arr_i +=1
            maxSum = max(maxSum, currSum)
        return maxSum
        """
        # not working

        # went out with the simple
        """
        def kadane(arr):
            currSum = maxSum = arr[0]
            for i in range(1, len(arr)):
                currSum = max(arr[i], currSum + arr[i])
                maxSum = max(maxSum, currSum)
            return maxSum

        n = len(nums)
        newNums = nums + nums[:-1]
        maxSum = float("-inf")
        for i in range(0,2*n - 1):
            currNums = newNums[i:n+i]
            maxSum = max(maxSum, kadane(currNums))
        
        return maxSum
        """

        # but this is O(n^2)

        # find maxSum of normal array
        # if it exists in the normal array, then thats fine
        # however, if it exists in the circular part
        # then, we need to check the max value including the circular part
        # however, if a max sum subarray exists in circular property, then the remaining elements form the
        # minimum sub array in that circular property

        # if we find minimum sum and subtract it from the total sum, it gives the max sum sub array
        # that includes the circular property.

        # for a circular array, maxSum + minSum should be equal to total sum
        # so, we need to find minSum
        # for that lets just negative of the array and find the maxSum again to get minSum

        def kadane(arr):
            maxSum = currSum = arr[0]
            for i in range(1, len(arr)):
                currSum = max(currSum + arr[i], arr[i])
                maxSum = max(maxSum, currSum)
            return maxSum

        maxSum = kadane(nums)

        negArray = [-num for num in nums]
        minSum = -1 * (kadane(negArray))

        totalSum = sum(nums)

        # however, we need to make sure that we are not picking the entire array
        # if minSum == totalSum, it implies we picked the entire array
        
        
        if minSum == totalSum:
            return maxSum

        maxSum_circular = totalSum - minSum

        return max(maxSum_circular, maxSum)
        


class Solution:
    def maxTurbulenceSize(self, arr: List[int]) -> int:

        # need to have a clever way that alternate sign is happening
        # lets say a represents i > i+1 and b represents i+1 < i+2 
        # we know that signs have alternated and we just need to propagate the value from a to b
        # and b to a as long as we find alternating
        # lets say we encountered element less than previous element
        # it implies less than operator
        # but if we use b to track if current element is greater than previous
        # it implies turulence has occured and we need to consider this length
        # we do this by propagation, we add to the flag which has occured and also add the previous flag to it
        # before resetting b, we propagate b value to a and add 1 to a
        #
        if not arr:
            return 0

        n = len(arr)
        dec = inc = 1
        max_len = 1

        for i in range(1,n):
            if arr[i-1] < arr[i]:
                inc = dec+1
                dec = 1
            elif arr[i-1] > arr[i]:
                dec = inc +1
                inc = 1
            else:
                inc = dec = 1
            max_len = max(max_len, inc,dec)

        return max_len


class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:

        """

        for L in range(len(nums)):
            # for R, we do from L to L+k+1
            for R in range(L+1, min(len(nums), L+k+1)):
                if nums[L] == nums[R]:
                    return True
        return False

        """

        

        window = set()
        L = 0

        for R in range(len(nums)):
            if R-L  > k :
                window.remove(nums[L])
                L+=1
            if nums[R] in window:
                return True

            window.add(nums[R])

        return False

class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:

        #
        
        if sum(nums) < target:
            return 0 

        min_length = float("inf")
        L = R =0
        currSum = 0

        for R in range(len(nums)):
            currSum += nums[R]

            while currSum >= target:
                min_length = min(min_length, R-L+1)
                currSum -= nums[L] 
                L +=1

        return min_length
        
class Solution:
    def numOfSubarrays(self, arr: List[int], k: int, threshold: int) -> int:
        
        # we need to find sub array of size k
        # we can maintain a set to store them, if it goes beyond set, we can remove them
        # and every time average goes above or beyond threshold, we increase the counter

        # sets are not suitable, as we are not looking for any dupliates or missing elements
        # we can just use a variable which tracks current sum and then divide it by k, to get current average

        """
        currentSum = 0
        L = 0
        counter = 0

        for R in range(len(arr)):
            if R-L + 1  > k :

                currentSum = currentSum - arr[L]
                L +=1

            currentSum += arr[R]

            if currentSum/k >= threshold:
                counter +=1
            
        return counter
        """

        # ok the question specifically asks for window size of k and the average not just either of them

        currentSum = 0
        counter = 0
        L =0

        for R in range(len(arr)):
            currentSum += arr[R]

            if R-L + 1 == k:
                if currentSum/k >= threshold:
                    counter += 1

                currentSum -= arr[L]
                L += 1

        return counter


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:

        # without repeating character
        # implies we always need to quickly check if we have already added
        # better to go with set

        """
        # set is appropriate but requires resetting
        window = set()
        result_string = {}
        max_len = 0
        output = ""
        count = 0
        L = 0
        for R in range(len(s)):
            l = s[R]

            if l in window:
                #result_string[count] = s[L:R+1]
                window = set()
                count = 0
                L = R # this doesnot work as we might be skipping few elements in betwee
                # which does not have duplicate
                # better way would to have map for each character
                # which gives index and we just right of this index
                

            if l not in window:
                window.add(l)
                count +=1
            max_len = max(max_len,count )

        #return result_string[max(result_string.keys())]
        return max_len

        """
        char_map = {}
        max_len = 0
        L =0

        for R in range(len(s)):
            if s[R] in char_map and char_map[s[R]] >= L:
                L = char_map[s[R]] +1
            char_map[s[R]] = R

            max_len = max(max_len, R-L+1)

        return max_len
    
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        
        """
        if not s:
            return 0
        max_len = 0
        #curr_len = 1
        L = 0
        counter = 0
        prev = s[0]
        for R in range(1,len(s)):
            if s[R-1] != s[R] and counter == k:
                L +=1
                counter =0
                prev = s[L]
                #curr_len = 1

            if s[R-1] != s[R] and prev == s[R-1] :
                counter += 1
                #curr_len += 1

            max_len = max(max_len, R-L+1)

        return max_len
        """

        # I was only checking one by one character and immediately resetting the counter
        # if it reaches K, but as this only looks for making changes in the consecutive elements
        # but, if we have non consectuive characters, then we might need to do changes some where in between
        # and it could effect the solution

        # instead, for any given string, if we are able to track the most frequent character
        # and count the number of changes to be made to make it further freuquent,
        # then we may find the longest uniform substring that obtained by replacing k times.

        max_freq = 0  #to find the max freq of each character
        char_count = {} #to store the count of each character
        max_len =0 

        L = 0

        for R in range(len(s)):
            char_count[s[R]] = char_count.get(s[R],0) + 1
            max_freq = max(max_freq, char_count[s[R]])

            while (R-L+1) - max_freq > k :
                char_count[s[L]] -=1
                L +=1
                if char_count[s[L]] < max_freq:
                    max_freq = max(char_count.values())

            max_len = max(max_len, R-L+1)

        return max_len


class Solution:
    def isPalindrome(self, s: str) -> bool:

        """
        filtered_chars = [char.lower() for char in s if char.isalnum()]
        clean_s = "".join(filtered_chars)
        L = 0
        R = len(clean_s)-1
        while L< R:
            if clean_s[L] != clean_s[R]:
                return False

            L +=1
            R -= 1
            
        return True
        """
        L,R = 0 , len(s) - 1
        while L < R:
            while L < R and not s[L].isalnum():
                L += 1
            while L < R and not s[R].isalnum():
                R -=1

            if s[L].lower() != s[R].lower():
                return False

            L+=1
            R -=1

        return True
        
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:

        """
        L = 0
        R = 0

        # lets use L to traverse
        #  we move R only to place unique elements
        prev = nums[0]
        for i in range(len(nums)):
            
            if prev != nums[i]:
                R += 1
                L = 0
            else:
                L += 1

            prev = nums[i]

        return R+1

        """

        if not nums:
            return 0

        L = 0

        for R in range(1, len(nums)):
            if nums[L] != nums[R]:
                 L +=1
                 nums[L] = nums[R]

        return L+1 
            
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:

        L = 1
        count = 1

        for R in range(1, len(nums)):
            if nums[R] == nums[R-1] :
                count += 1
            else:
                count = 1

            if count <= 2:
                nums[L] = nums[R]
                L +=1
               

        return L
    

class Solution:
    def maxArea(self, height: List[int]) -> int:
        
        # the thing is we can do brute force and search for best combination and return them
        # it will O(n^2)

        # also, the area is not just multiplication
        # its getting the height indicated by array values, infact its the minimum of those heights
        # and getting the distance between them

        # and we already a width dimension, along which we can try to maximize the area by 
        # chosing the wide pointers, heights at the edges
        # now question becomes how to traverse from L,R at the initial left and right pointers
        # the thing is we should not reduce width for maximum area
        # so, of both the heights that is possible, we will retain the pointer which has max h value
        # so that we can have least reduction in area by reducing width with the contraction.

        if not height :
            return 0

        L = 0
        R =len(height) - 1
        max_area = 0

        while L < R :
            h1 = height[L]
            h2 = height[R]
            area = min(h1, h2)*abs(R-L)
            max_area = max(max_area,area)

            if h1 > h2:
                R -= 1
            elif h1 < h2:
                L +=1
            else:
                L +=1
                R -= 1

        return max_area


class Solution:
    def trap(self, height: List[int]) -> int:

        # at any bar, or position, the max water that can be stored at the position
        # would be [3,4,2] - no water no where
        # for [3,1,2] - no water at 3,2. at 1 height would (3-2) - 1 as the water is its above
        # for [3,0,2,4,1,0] # we do not know left and right max dirfectly
        # ofcourse we can know, like, we need to know left and right max for each bar
        # for each bar, we can scan left and right side of each time and calculate it
        # otherwise, we can precompute and store left max and right max arrays and do it
        3 

        if not height :
            return 0

        L, R = 0, len(height) - 1
        left_max = height[0]
        right_max = height[-1]
        water = 0

        # water trapped at any point is dependent on not only the adjacent elements
        # as we can see from the given the given example, water trapped at 0 not dependent between 
        # on the adjacent height, but is dependent on the max height from either height seen fo far
        # moreover we do not fill each bar in single direction

        
        # Water trapping is not determined by adjacent bars alone but by the highest bars
        # to the left and right of each position across the entire array.
        # This process requires knowing the maximum height encountered so far from both directions,
        # which we track with left_max and right_max.

        # The algorithm oscillates between moving the left or right pointer inward,
        # depending on which side currently has the lesser boundary height. This strategy
        # ensures that the calculation for trapped water is conservative and accurate because
        # it relies on the confirmed shorter side of the boundary to determine potential water trapping.


        while L < R:
            if height[L] <= height[R]:
                # Move the left pointer when the left side is lower or equal.
                # This approach guarantees that the right_max is a reliable boundary because
                # it is either higher or equal to the left_max, thus safely containing the water above the current left bar.
                if height[L] >= left_max: # if the current height is greate than left_max so far,
                    left_max = height[L]  # no water gets trapped
                else:
                    water += left_max - height[L] #otherwise water gets trapped
                L+=1

            else: 
                if height[R] >= right_max:
                    right_max = height[R]
                else:
                    water += right_max - height[R]
                R -=1

        return water
    

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def pairSum(self, head: Optional[ListNode]) -> int:

        # find mid point
        # then reverse the linked list
        # 
        fast = head
        slow = head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        midpoint = slow
        # how to reverse a linked list
        # need to reverse
        prev = None
        current = midpoint
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node

        # prev
        max_sum = 0
        first_half = head
        second_half = prev

        while first_half and second_half :
             max_sum = max(max_sum, first_half.val + second_half.val)
             first_half = first_half.next
             second_half = second_half.next

        return max_sum


        # head becomes tail and tail becomes head
        

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return None 
        fast = slow = head

        # lets assume a distance of D from head to start of the cycle
        # lets assume fast and slow meet at P from start of the cycle
        # and lets assume C to be the length of the cycle
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                #break

        # slow is currently at P and has travelled D+P distance
        # since fast travels twice as fast as slow
        # it must have travelled 2 (D+P) distance
        # however, since it is moving in cycles
        # we can say, it has travelled D + P + some multiple of C distance
        # D+P + nC = 2*(D+P), D+ P = nC
        # for n = 1, D+ P = C, and if slow is already at P from the start of the cycle
        # it just has to travel D distance to reach the start.
        # and we know that D is the distance from head to start of the cycle
        # since we do not know D, lets consider another pointer starting from head
        # and move both slow2 and slow each step until they meet
        # the point they meet should be start of the cycle
        
        #if not fast or not fast.next:
        #    return None
                slow2  = head
                while slow and slow2 :
                    if slow == slow2:
                        return slow
                    slow = slow.next
                    slow2 = slow2.next
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        # duplicate number implies cycle
        # we can do this sets, but it costs, O(n) space
        # they say it should be in O(1) space

        """
        fast = slow = nums[0]
        i = 0
        while i < len(nums)-2:
            fast = nums[i+2]
            slow = nums[i+1]
            i+=1
            if fast == slow:
                return slow

        """

        # nums contains n + 1 integers where each integer is in the range [1, n] inclusive.
        # so we can treat  values as integers and we can keep traversing the array
        # and we wont go out of index
        # and since we take values as index, we have a chance that both our pointers meet at some point


        # The presence of a duplicate means that at least one number in the range from 1 to n appears 
        # more than once. This duplicate creates a cycle because it will index back to an already accessed
        # index at some point, as you follow from index to index using the values as pointers.

        fast = nums[nums[0]]
        slow = nums[0]
        while slow != fast:
            slow = nums[slow]
            fast = nums[nums[fast]]
            # however there is no guarantee that the point they meet is where the duplicate is
            # the presence of duplicate led to move them in circles
            # but there is no guarantee that they meet at the duplicate
            # for e.g. [1,3,4,2,2] in this case both the duplicates 2,2 point to the same index 3

            # fast = 3, slow = 1
            # fast = nums[nums[3]] = nums[2] = 4, slow = nums[1] = 3
            # fast = nums[nums[4]] = nums[2] = 4, slow = nums[3] = 2
            # fast = nums[nums[4]] = nums[2] = 4, slow = nums[2] = 4 # but 4 is not the duplicate here.
            # 2 is the duplicate 
            # in normal case this would have worked, but in this case, they are pointing to themselves
            # that is fast = nums[nums[4]] = nums[2] = 4, slow = nums[4] = 2
            # fast is just pointing to itself and slow is stuck in a loop


            # we just know that there is a cycle
            # but we dont know if its the duplicate

            # The duplicate value 2 leads to a cycle because it occurs more than once, 
            # and both occurrences point back to the same next index (index 2), creating a cycle.
            # The pointers do not need to visit every instance of the duplicate value directly; 
            # instead, they need to detect the cycle caused by the repeated index access which is the signature of a duplicate.

        slow2 = 0
        while slow != slow2:
            slow2 = nums[slow2]
            slow = nums[slow]

        return slow
        

"""
   
                
anyways, looks like to search across a grid, dfs is inevitable, but to effectively reduce paths in search, we are using prefix tree (Trie) to avoid non essential paths ?
ChatGPT
Yes, you've summarized the essence of using a trie in the word search problem quite accurately. DFS is indeed a common approach for exploring all possible paths in a grid-based puzzle like this, and the trie (prefix tree) is employed to efficiently reduce the number of paths explored by cutting off those that cannot lead to valid words early in the search process. This combination allows for deep exploration while maintaining efficiency by pruning non-viable options.

Alternatives to Using Trie and Why Trie is Preferable:
Let's explore other potential methods and compare why a trie might be favored over alternatives like hashmaps in this specific context:

Using Hashmaps:
Method: Store each word in a hashmap and as you explore each path in the DFS, construct the word and check if it exists in the hashmap.
Limitations:
Space Complexity: While a hashmap provides constant time complexity for lookups, storing every possible prefix of each word (necessary for effective pruning during DFS) could significantly increase the memory usage.
Efficiency in Pruning: Unlike tries, hashmaps do not inherently support prefix-based pruning. To achieve similar pruning, you'd need to store all possible prefixes of all words, which isn't space-efficient.
Using HashSet for Words and Prefixes:
Method: Similar to using a hashmap but involves storing all words and their prefixes in a set for O(1) access.
Limitations:
Memory Usage: You end up storing a large number of prefixes, especially for longer words, which can be memory-intensive.
Construction Overhead: Every insertion requires adding not only the word but all its prefixes to the set, increasing the setup time.
Bloom Filters:
Method: A space-efficient probabilistic data structure that can be used to test whether an element is a member of a set, suitable for situations where some false positives are tolerable.
Limitations:
False Positives: While bloom filters are space-efficient, they can yield false positives, which could lead to unnecessary exploration in DFS.
No Prefix Support: Like hashmaps, they don’t inherently support prefix checks unless combined with other structures.
Why Trie is Preferable:
Prefix Efficiency: Tries provide an inherent advantage in prefix-based searches, which is crucial in this problem. They allow for checking and pruning paths based on prefixes dynamically as the search progresses.
Space Efficiency for Prefixes: While tries do consume space, they store common prefixes once, which is typically more space-efficient than storing all prefixes separately in a hashmap or set.
Direct Support for Word Completion: Tries not only manage prefixes efficiently but also seamlessly handle the check for complete words by simply marking nodes as word nodes.
DFS and Trie Working Together:
Optimal Pruning: As you traverse the grid with DFS, each step checks against the trie. If no children nodes correspond to the next board character, the path is pruned immediately.
Efficient Backtracking: Trie helps ensure that each recursive call in the DFS has the potential to lead to a valid word, thus making the backtracking that does occur meaningful and necessary.

"""

class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        curr = self.root
        for c in word:
            if c not in curr.children:
                curr.children[c] = TrieNode()
            curr = curr.children[c]
        
        curr.word = word

    def search(self, word):
        curr = self.root
        for c in word:
            if c not in curr.children:
                return False
            curr = curr.children[c]
        return curr.word


class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:

        trie = Trie()
        for word in words:
            trie.insert(word)

        result = set() # to prevent adding the same words founds through different paths

        rows, columns = len(board), len(board[0])

        # lets define dfs
        def dfs(r,c,node, visited):

            if (r< 0 or r >= rows or c <0 or c >= columns or
                (r,c) in visited or board[r][c] not in node.children):
                return 

            visited.add((r,c))
            node = node.children[board[r][c]]

            if node.word:
                result.add(node.word)
                node.word = None
            dfs(r+1,c, node,visited)
            dfs(r,c+1, node, visited)
            dfs(r-1, c, node, visited)
            dfs(r,c-1, node, visited)

            visited.remove((r,c))



        # we need to search words, that can emanate from each cell and the 4 paths from it
        for r in range(rows):
            for c in range(columns):
                dfs(r,c,trie.root, set()) # for each combinattion of r,c, we initiate a visited set as we we cannot use the same cell twice

        

        return list(result)
    

"""
Time Complexity:
DFS Complexity : we initate a DFS from each cell and from there 4 other dfs calls, so, 4^m*n or
but due to back tracking and trie based pruning not all paths are explored, so 4^L, where L is the length of the largest word
Trie Look up : O(1) for each character for each word at any point we only check one point
Total = O(m*n*4^L), where m and n are rows and columns
Space Complexity:
Trie Storage: O(n*m), where n is the number of words and m is the avg length of the words
DFS Stack: we need to remember all the recursion variable for each recursion call and this depends on the size of the longest word
O(L)
Visited Set : each dfs call, creates a visited set to store cells, to track characters and it depends on the longest word.
So, O(L)
Total = O(n*m + L)

"""

class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = False

class WordDictionary:
    def __init__(self):
        self.root = TrieNode()
        

    def addWord(self, word: str) -> None:
        curr = self.root
        for c in word:
            if c not in curr.children:
                curr.children[c] = TrieNode()
            curr = curr.children[c]
        curr.word = True
        
    """
    def search(self, word: str) -> bool:
        curr = self.root
        for c in word:
            if c not in curr.children:
                return False
            curr = curr.children[c]
        return curr.word
    """

    def search(self, word) :
        return self._search(word, 0, self.root)


    def _search(self, word, index, node):
        if index == len(word): #if no dots, then simply, if you were able to reach the end, then return that the word is found
            return node.word  # works for both words that does not have any "." and also for words that have "." to search sequences later.

        char = word[index]
        if char == ".":

            # search all the paths of all children from here
            # we m
            for child in node.children.values():
                if self._search(word, index+1, child):
                    return True
            return False

        else:
            if char not in node.children:
                return False
            return self._search(word, index+1, node.children[char])
        


class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        # use two pointer fast and slow
        slow = fast =head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
    


class WordFilter:

    def __init__(self, words: List[str]):

        # use a dictionary to store index value for each prefix and suffix possible
        # and we combine them using a separator
        # using a separatot helps avoid ambiguities
        # Consider a scenario with a prefix "ab" and a suffix "cd". 
        # Without a separator, the key becomes "abcd". 
        # This could also be interpreted as a prefix "abc" and a suffix "d", 
        # or any other valid combination that forms "abcd".

        self.lookup = {}
        for index, word in enumerate(words):
            length = len(word)
            for i in range(length+1):
                for j in range(length + 1) : # +1 to account for empty suffix and prefixes
                    prefix = word[:i]
                    suffix = word[-j:] if j != 0 else ""
                    key = prefix + "#" + suffix

                    if key in self.lookup:
                        # Always store the maximum index for this prefix-suffix pair
                        self.lookup[key] = max(self.lookup[key], index)
                    else:
                        self.lookup[key] = index

                    #self.lookup[prefix + "#" + suffix] = index
        

    def f(self, pref: str, suff: str) -> int:
        key = pref + "#" + suff
        return self.lookup.get(key,-1)        
    

    # Time Complexity = O(n*k^2) where n is the number of words and k is the length of the longest word
    # Space Complexity is similar to construct hashmap for n*k^2 combinations
    # but this was manageable as suff and pref can only be maximum of 7 characters and we have a total of 14 characters at the max.


"""
class WordFilter:

    def __init__(self, words: List[str]):

        # use a dictionary to store index value for each prefix and suffix possible
        # and we combine them using a separator
        # using a separator helps avoid ambiguities
        # Consider a scenario with a prefix "ab" and a suffix "cd". 
        # Without a separator, the key becomes "abcd". 
        # This could also be interpreted as a prefix "abc" and a suffix "d", 
        # or any other valid combination that forms "abcd".

        self.lookup = {}
        for index, word in enumerate(words):
            length = len(word)
            for i in range(length+1):
                for j in range(length + 1) : # +1 to account for empty suffix and prefixes
                    prefix = word[:i]
                    suffix = word[-j:] if j != 0 else ""
                    key = prefix + "#" + suffix

                    if key in self.lookup:
                        # Always store the maximum index for this prefix-suffix pair
                        self.lookup[key] = max(self.lookup[key], index)
                    else:
                        self.lookup[key] = index

                    #self.lookup[prefix + "#" + suffix] = index
        

    def f(self, pref: str, suff: str) -> int:
        key = pref + "#" + suff
        return self.lookup.get(key,-1)        
"""

    # Time Complexity = O(n*k^2) where n is the number of words and k is the length of the longest word
    # Space Complexity is similar to construct hashmap for n*k^2 combinations
    # but this was manageable as suff and pref can only be maximum of 7 characters and we have a total of 14 characters at the max.



# generally we use tree nodes for search, and just to check if the word ends at the current character
# by adding a trienode at every character in the word

# but this problem asks for indices, so we are better of storing the indices of the word, at each character node in the word
# so that once the word is found, we can return the max of it

# we use both prefix and suffix trees to reduce the space complexity
# search time complexity is O(1) for each character and it will then be O(m)
# space complexity is O(m*n) where m is average length

"""
class TrieNode:
    def __init__(self):
        self.children = {}
        #self.index = set()
        self.index = []

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, index):
        curr = self.root
        for c in word:
            if c not in curr.children:
                curr.children[c] = TrieNode()
            curr = curr.children[c]
            #curr.index.add(index)
            curr.index.append(index)
    def search(self, word):
        curr = self.root
        for c in word:
            if c not in curr.children:
                # return set()
                return []
            curr = curr.children[c]
        return curr.index

class WordFilter:

    def __init__(self, words: List[str]):
        # create prefix and suffix tree
        self.prefixTree = Trie()
        self.suffixTree = Trie()

        for index, word in enumerate(words):
            self.prefixTree.insert(word, index)
            self.suffixTree.insert(word[::-1], index)


    def f(self, pref: str, suff: str) -> int:

        pref_index = self.prefixTree.search(pref)
        suff_index = self.suffixTree.search(suff[::-1])

        if pref_index == [] or suff_index == []:
            return -1

        #if pref_index is None or suff_index is None:
        #if not pref_index or not suff_index :
        #    return -1

        result = -1
        pref_index_set = set(pref_index)
        for index in suff_index:
            if index in pref_index_set:
                result = max(result, index)

        
        common_index = pref_index & suff_index
        if common_index : 
            return max(common_index)
        return -1
        
        return result

"""

# as data is naturally sorted, I mean we insert the words in a sequential order one word after word
# so, last word will have the largest index, and it makes sense to search from the back side
# instead of converting to sets, everytime, as they say about 10^4 calls will be made.
class TrieNode:
    def __init__(self):
        self.children = {}
        self.indices = []  # Stores indices of words that reach this node

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, index):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.indices.append(index)  # Append index at every node down the path

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return []
            node = node.children[char]
        return node.indices



class WordFilter:
    def __init__(self, words):
        self.prefix_trie = Trie()
        self.suffix_trie = Trie()
        for index, word in enumerate(words):
            self.prefix_trie.insert(word, index)  # Insert normally for prefix
            self.suffix_trie.insert(word[::-1], index)  # Insert reversed for suffix

    def f(self, pref, suff):
        pref_indices = self.prefix_trie.search(pref)
        suff_indices = self.suffix_trie.search(suff[::-1])  # Reverse the suffix to search in suffix trie
        
        if suff_indices == [] or pref_indices == []:
            return -1
        
        i,j = len(pref_indices)-1, len(suff_indices)-1
        while i >= 0 and j >= 0:
            if pref_indices[i] == suff_indices[j]:
                return pref_indices[i]
            if pref_indices[i] > suff_indices[j]:
                i -= 1
            else:
                j -= 1
        return -1
        
        
        
        """
        # Find the maximum index present in both lists
        result = -1
        pref_set = set(pref_indices)
        for index in suff_indices:
            if index in pref_set:
                result = max(result, index)
        return result

        """




# Your WordFilter object will be instantiated and called as such:
# obj = WordFilter(words)
# param_1 = obj.f(pref,suff)