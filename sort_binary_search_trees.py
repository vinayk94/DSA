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






