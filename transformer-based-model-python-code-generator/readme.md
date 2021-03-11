# Transformer Based Model Python Code Generator

Capstone project is to write a transformer-based model that can write python code (with proper whitespace indentations).



## Dataset

You can find the dataset [here (Links to an external site.)](https://drive.google.com/file/d/1rHb0FQ5z5ZpaY2HpyFGY6CeyDG0kTLoO/view?usp=sharing). There are some 4600+ examples of English text to python code. 



## Data Preproessing



## Model Architecture and Salient Features



## Data Preparation Strategy



## Embedding Strategy



## Metrices



## Output from Mode



```
Question: write a python function to return the content of a directory and the last modified date
Source Python:
import glob 
 import os 
 import time 
 def retrieve_files_bydate(src_dir_path,*args ) : 
     if(os.path.exists(src_dir_path ) = = false ) : 
         print("destination path does n't exist " ) 
         return 
     files_in_dir = glob.glob(src_dir_path+"/ * . * " ) 
     if ( len(files_in_dir ) < = 0 ) : 
         print("no files present in:",src_dir_path ) 
         return 
     file_date_list = [ ( filename , time.ctime(os.path.getmtime(filename)))for filename in files_in_dir ] 
     return file_date_list


Target Python:
import math 
 def <unk> ) : 
     if(os.path.exists(src_dir_path ) = = false ) : 
         print("destination path does n't exist " ) 
         return 
     files_in_dir = glob.glob(src_dir_path+"/ * . * . * . * . * . * . * . * . 
#########################################################################################################
#########################################################################################################
Question: write a python function that takes a list of words and returns the longest one
Source Python:
def find_longest_word(words_list ) : 
 word_len = [ ] 
 for n in words_list : 
 word_len.append((len(n ) , n ) ) 
 word_len.sort ( ) 
 return word_len[-1][1 ] 
 print(find_longest_word(["php " , " python " , " zekelabs " ] ) )


Target Python:
def printwords(s ) : 
     s = s.split ( ' ' ' ' ) 
     for word in s : 
         if len(word)%2==0 : 
             print(word ) 
#########################################################################################################
#########################################################################################################
Question: split strings
Source Python:
word = " hello world " 
 ksplit = word.split ( ' ' ) 
 print(f"splited strings : { ksplit } " )


Target Python:
word = " hello world " 
 ksplit = word.split ( ' ) 
 print(f"splited strings : { ksplit } " ) 
#########################################################################################################
#########################################################################################################
Question: write a python function to find greatest common divisor
Source Python:
def greatest_common_divisor(x , y ) : 
     print("for " , x , " and " , y , " , " ) 
     r = x%y 
     while r>0 : 
         r = x%y 
         if r = = 0 : 
             print("the greatest common divisor is " , y , " . " ) 
         else : 
             q = y 
             x = q 
             y = r 
 greatest_common_divisor(1071,1029 )


Target Python:
def lcm(num1 , num2 ) : 
     if num1 > num2 : 
         return true 
     else : 
         return false 
#########################################################################################################
#########################################################################################################
Question: write a python program to make use of maps
Source Python:
def square(number ) : 
     return number * * 2 
 numbers = [ 1 , 2 , 3 , 4 , 5 ] 
 squared = map(square , numbers ) 
 print(f'mapped numbers:{list(squared ) } ' )


Target Python:
<unk> = ' <unk> ' 
 <unk> ) 
#########################################################################################################
#########################################################################################################
Question: write a function to print if a number is even or odd
Source Python:
def oddeven(num ) : 
     if num % 2 = = 0 : 
         print('even ' ) 
     else : 
         print('odd ' )


Target Python:
def <unk> ) : 
     if num > 0 : 
         return num 
     else : 
         return num % 10 
     for i in range(2 , num ) : 
         if num % i ) = = 0 : 
             if num % i ) = = 0 :
#########################################################################################################
#########################################################################################################
Question: write a python function to find the number of ( i , j ) pairs where i < j and ar[i]+ar[j ] is divisible by k in a data list
Source Python:
def divisible_sum_pairs(arr , k ) : 
     count = 0 
     n = len(arr ) 
     for i in range(n - 1 ) : 
         j = i + 1 
         while j < n : 
             if ( ( arr[i ] + arr[j ] ) % k ) = = 0 : 
                 count + = 1 
             j + = 1 
     return count 
 import math


Target Python:
def <unk> , k ) : 
     count = 0 
     for i in range(n ) : 
         j = i + 1 
         <unk> ] + = k 
     return count 
#########################################################################################################
#########################################################################################################
Question: write a python function to remove leading zeros from an ip address
Source Python:
import re 
 regex = ' \.[0 ] * ' 
 def remove_leading_zeros(ip ) : 
     modified_ip = re.sub(regex , ' . ' , ip ) 
     return modified_ip


Target Python:
import re 
 def <unk> ) : 
     regex = ' <unk> ] 
     return re.sub('[^a - za - z0 - <unk> ' , s ) 
#########################################################################################################
#########################################################################################################
Question: write a function to count the number of digits in a number
Source Python:
def count_digits(n ) : 
     return len(str(n ) )


Target Python:
def <unk> ) : 
     n = 0 
     while n > 0 : 
         n = n = n % 10 
         n = n = n = n // 10 
     return n 
#########################################################################################################
#########################################################################################################
Question: write a function to return the total surface area of a cube of side a
Source Python:
def cal_surface_area_cube(a ) : 
     return 6*(a**2 )


Target Python:
def cal_surface_area_cube(a ) : 
     return 6*(a**2 ) 
#########################################################################################################
#########################################################################################################
Question: write a python program to implement linear search and print the key element if found
Source Python:
def linear_search(alist , key ) : 
     " " " return index of key in alist . return -1 if key not present . " " " 
     for i in range(len(alist ) ) : 
         if alist[i ] = = key : 
             return i 
     return -1 


 alist = [ 2 , 3 , 5 , 6 , 4 , 5 ] 
 key = 6 

 index = linear_search(alist , key ) 
 if index < 0 : 
     print(f'{key } was not found . ' ) 
 else : 
     print(f'{key } was found at index { index } . ' )


Target Python:
def linear_search(alist , key ) : 
     if key in range(len(alist ) : 
         return -1 
     return -1 
     return -1 


 alist = [ 2 , 3 , 4 ] 
 index = [ 6 , 5 , 6 , 7 , 7 ] 
 index = [
#########################################################################################################
#########################################################################################################
Question: write a function that removes all special characters
Source Python:
def clean_str(s ) : 
     import re 
     return re.sub('[^a - za - z0 - 9]+ ' , '' , s )


Target Python:
def <unk> ) : 
     import re 
     return re.sub('[^a - za - z0 - <unk> ' , s ) 
#########################################################################################################
#########################################################################################################
Question: function to add two tuple
Source Python:
def add_tuple(tup1 , tup2 ) : 
     return tup1+tup2


Target Python:
def <unk> , b ) : 
     return ( a , b ) ) 
#########################################################################################################
#########################################################################################################
Question: function to print ascii value of a character .
Source Python:
def show_ascii(a : str ) : 
     print(ord(a ) )


Target Python:
def <unk> ) : 
   return <unk> ) 
#########################################################################################################
#########################################################################################################
Question: write a function to calculate the potential energy of an object of mass m at height h
Source Python:
def cal_pe(mass : float , height : float)->float : 
     g = 9.8 
     return ( mass*g*height )


Target Python:
def <unk> : float , height : float)->float : 
     return ( <unk> ) 
#########################################################################################################
#########################################################################################################
Question: n⋅2n − 1 , with n ≥ 1 .
Source Python:
def woodall_number(n ) : 
     if n > = 0 : 
         return n * 2 * * n - 1


Target Python:
def <unk> ) : 
     if n > = 0 : 
         return 0 
         return <unk> ) + <unk> ) + <unk> ) 
#########################################################################################################
#########################################################################################################
Question: write a function to return the area of a trapezium with base a base b and height h between parallel sides
Source Python:
def cal_area_trapezium(a , b , h ) : 
     return h*(a+b)/2


Target Python:
def cal_area_trapezium(a , b , h ) : 
     return h*(a+b)/2 
#########################################################################################################
#########################################################################################################
Question: write a lambda function that gives true if the input number is even otherwise false
Source Python:
even = lambda a : true if a%2 = = 0 else false


Target Python:
<unk> = <unk> ) 
#########################################################################################################
#########################################################################################################
Question: write a python function to calculate simple interest
Source Python:
def simple_interest(p , t , r ) : 

     si = ( p * t * r)/100 
     return si


Target Python:
def simple_interest(p , r , t , t , t ) : 
     si = ( p * t * t * r)/100 
     return si 
#########################################################################################################
#########################################################################################################
Question: iterate through rows
Source Python:
for i in range(len(x ) ) : 
    # iterate through columns 
    for j in range(len(x[0 ] ) ) : 
        result[i][j ] = x[i][j ] + y[i][j ] 
 for r in result : 
    print(r )


Target Python:
for i in range(len(x ) ) : 
    # iterate through columns 
    for j in range(len(x[0 ] ) ) : 
        result[j][i ] = x[i][j ] 
 for r in result : 
    print(r ) 
#########################################################################################################
#########################################################################################################
Question: write a python function to get the surface_area of a cone with radius & slant height as input
Source Python:
def cone_surface_area(radius , slant_height ) : 
     surface_area =   3.14 * ( radius * * 2 ) + 3.14 * radius * slant_height 
     return surface_area


Target Python:
def <unk> , height ) : 
     surface_area = <unk> + ( <unk> ) + ( <unk> ) + ( <unk> ) ) * 3.14 * 3.14 ) 
     return surface_area 
#########################################################################################################
#########################################################################################################
Question: write a python function to read a csv file and print its content
Source Python:
def read_csv(filename ) : 
     import csv 
     with open(filename , newline= ' ' ) as f : 
         reader = csv.reader(f ) 
         for row in reader : 
             print(row )


Target Python:
def <unk> ) : 
     import csv 
     with open(filename , newline= ' ' ' ' ' ) as f : 
         reader = <unk> ) 
         for row in reader : 
             print(row ) 
#########################################################################################################
#########################################################################################################
Question: usage of dictionary
Source Python:
dict = { ' name ' : ' zara ' , ' age ' : 7 , ' class ' : ' first ' } 
 print " dict['name ' ] : " , dict['name ' ] 
 print " dict['age ' ] : " , dict['age ' ]


Target Python:
k = 2 
 for i in <unk> ) : 
     if i = = 1 : 
         <unk> ) 
 <unk> ) 
#########################################################################################################
#########################################################################################################
Question: given a two list of equal size create a set such that it shows the element from both lists in the pair
Source Python:
firstlist = [ 2 , 3 , 4 , 5 , 6 , 7 , 8 ] 
 secondlist = [ 4 , 9 , 16 , 25 , 36 , 49 , 64 ] 
 result = zip(firstlist , secondlist ) 
 resultset = set(result ) 
 print(resultset )


Target Python:
firstlist = [ 2 , 3 , 4 , 5 , 6 , 7 , 8 ] 
 secondlist = [ 4 , 9 , 16 , 25 , 36 , 49 , 64 ] 
 result = zip(firstlist , secondlist ) 
 resultset = set(result ) 
 print(resultset )
#########################################################################################################
#########################################################################################################
Question: write a python function get the maximum number in passed list
Source Python:
def max_check(x ) : 
     max_val = x[0 ] 
     for check in x : 
         if check > max_val : 
             max_val = check 
     return max_val 
 print(f'{max_check([2,4,5,7,98 ] ) } ' )


Target Python:
def <unk> ) : 
     min_val = x[0 ] 
     for check in lst : 
         min_val = check 
     return min_val 
 <unk> ] ) 
#########################################################################################################
#########################################################################################################
```

