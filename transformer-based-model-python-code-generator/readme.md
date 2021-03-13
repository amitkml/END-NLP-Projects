# Transformer Based Model Python Code Generator

Capstone project is to write a transformer-based model that can write python code (with proper whitespace indentations).

## What is Transformer Network?

The transformer model is based entirely on the attention mechanism and completely gets
rid of recurrence. The transformer uses a special type of attention mechanism called self-attention.

The transformer consists of an encoder-decoder architecture. We feed the input sentence (source
sentence) to the encoder. The encoder learns the representation of the input sentence and
sends the representation to the decoder. The decoder receives the representation learned by
the encoder as input and generates the output sentence (target sentence).

To process a sentence we need these 3 steps:

1. Word embeddings of the input sentence are computed simultaneously.
2. Positional encodings are then applied to each embedding resulting in word vectors that also include positional information.
3. The word vectors are passed to the first encoder block.

## Transformer Encoder



Each block consists of the following layers in the same order:

1. A multi-head self-attention layer to find correlations between each word
2. A [normalization](https://theaisummer.com/normalization/) layer
3. A residual connection around the previous two sublayers
4. A linear layer
5. A second normalization layer
6. A second residual connection

![im](https://theaisummer.com/assets/img/posts/transformer/encoder.png)

### Self-attention mechanism

Consider the following sentence: A dog ate the meat because it was hungry

By reading the above statement, we can easily understand pronoun “it” relates to “dog” rather that meat. But how a model automatically understand this?

Model takes representation of each such word and then relate with all order it to understand which other words are strongly related to it. So, that’s how it will understand.

## Transformer decoder

1. The output sequence is fed in its entirety and word embeddings are computed
2. Positional encoding are again applied
3. And the vectors are passed to the first Decoder block

Each decoder block includes:

1. A **Masked** multi-head self-attention layer
2. A normalization layer followed by a residual connection
3. A new multi-head attention layer (known as **Encoder-Decoder attention**)
4. A second normalization layer and a residual connection
5. A linear layer and a third residual connection

![im](https://theaisummer.com/assets/img/posts/transformer/decoder.png)

## Output of Transformer

The output probabilities predict the next token in the output sentence. How? In essence, we assign a probability to each word in the French language and we simply keep the one with the highest score.



## Dataset

You can find the dataset [here (Links to an external site.)](https://drive.google.com/file/d/1rHb0FQ5z5ZpaY2HpyFGY6CeyDG0kTLoO/view?usp=sharing). There are some 4600+ examples of English text to python code. 



## Data Preproessing



## Model Architecture and Salient Features



## Data Preparation Strategy



## Embedding Strategy



## Metrices



## Final Model



## Experimentations during Project

I have done several experimentation during my capstone projects and have summarized below my experimentation.

Key changes done in model architecture

### Model with TEXT.build_vocab(train_data, min_freq = 2)



### Model with TEXT.build_vocab(train_data, min_freq = 1)



### Model with TEXT.build_vocab(train_data, min_freq = 1) and higher size of hidden layer

### Model with TEXT.build_vocab(train_data, min_freq = 1) and custom tokenizer to handle python special charecters

Model Encoder and Decoder dimension has been increased from 512 to 1024

```
INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = len(TEXT.vocab)
HID_DIM = 512
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 1024
DEC_PF_DIM = 1024
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
```

Model Performance

```
| Test Loss: 2.166 | Test PPL:   8.723 |
```

Model Prediction on Test Data

```
Question: Write a function to calculate Volume of Hexagonal Pyramid
Source Python:
def volumeHexagonal(a , b , h ) : 
     return a * b * h


Target Python:
def cal_dist_from_orign(x : float , y : float)->float : 
     return ( x**2+y**2)**(1/2 ) 
#########################################################################################################
#########################################################################################################
Question: write a python program to multiply two list with list comprehensive
Source Python:
l1=[1,2,3 ] 
 l2=[4,5,6 ] 
 print([x*y for x in l1 for y in l2 ] )


Target Python:
test_list = [ ( 1 , 5 ) , ( 1 ) , ( 3 ) , ( 1 ) , ( 3 ) ] 
 res = [ ] 
 for i , ( i ) for i , j in test_list : 
     for i ) ] )
#########################################################################################################
#########################################################################################################
Question: Write a Python Program to Find the Intersection of Two Lists
Source Python:
def main(alist , blist ) : 
     def intersection(a , b ) : 
         return list(set(a ) & set(b ) ) 
     return intersection(alist , blist )


Target Python:
test_list = [ ( 1 , 5 ) , ( 1 ) , ( 3 ) , ( 1 ) , ( 3 ) ] 
 res = [ ] 
 for i , ( i ) for i , j in test_list : 
     for i ) ] )
#########################################################################################################
#########################################################################################################
Question: write a program to print count of number of unique matching characters in a pair of strings
Source Python:
str1="ababccd12@ " 
 str2="bb123cca1@ " 
 matched_chars = set(str1 ) & set(str2 ) 
 print("No . of matching characters are : " + str(len(matched_chars ) ) )


Target Python:
str1 = " Hello ! " 
 print("\"%s\ " 
 print('"%s " 
 print('"%s " 
 print('"{}"'.format(str1 ) ) ) 
#########################################################################################################
#########################################################################################################
Question: Write a function to calculate median of a list of numbers given
Source Python:
def median(pool ) : 
     copy = sorted(pool ) 
     size = len(copy ) 
     if size % 2 = = 1 : 
         return copy[int((size - 1 ) / 2 ) ] 
     else : 
         return ( copy[int(size/2 - 1 ) ] + copy[int(size/2 ) ] ) / 2


Target Python:
def shift_and_scale(list_of_nums , mean , std ) : 
     return [ ( x - mean ) / std for x in list_of_nums ] 
#########################################################################################################
#########################################################################################################
Question: Given a list slice it into a 3 equal chunks and revert each list
Source Python:
sampleList = [ 11 , 45 , 8 , 23 , 14 , 12 , 78 , 45 , 89 ] 
 length = len(sampleList ) 
 chunkSize   = int(length/3 ) 
 start = 0 
 end = chunkSize 
 for i in range(1 , 4 , 1 ) : 
   indexes = slice(start , end , 1 ) 
   listChunk = sampleList[indexes ] 
   mylist = [ i for i in listChunk ] 
   print("After reversing it " , mylist ) 
   start = end 
   if(i ! = 2 ) : 
     end + = chunkSize 
   else : 
     end + = length - chunkSize


Target Python:
def Sort_Tuple(tup ) : 

     lst = len(tup ) 
     for i in range(0 , lst ) : 
         for j in range(0 , lst ) : 
             if ( tup[j][1 ] > tup[j + 1][1 ] ) : 
                 tup[j + 1]= temp 
     return tup 
#########################################################################################################
#########################################################################################################
Question: write a python function to count ' a 's in the repetition of a given string ' n ' times .
Source Python:
def repeated_string(s , n ) : 
     return s.count('a ' ) * ( n // len(s ) ) + s[:n % len(s)].count('a ' )


Target Python:
def check(string ) : 
     s = False 
     p = False 
     if s = = = False : 
         return True 
     else : 
         return False 
#########################################################################################################
#########################################################################################################
Question: Write a Python Program to Check if a Number is a Perfect Number
Source Python:
def perfect_no_check(n ) : 
     sum1 = 0 
     for i in range(1 , n ) : 
         if(n % i = = 0 ) : 
             sum1 = sum1 + i 
     if ( sum1 = = n ) : 
         return True 
     else : 
         return False


Target Python:
num = 16 
 if num < 0 : 
    print("Enter a positive number " ) 
 else : 
    sum = 0 
    # use while loop to iterate until zero 
    while(num > 0 ) : 
        sum + = num -= 1 
    print("The sum is "
#########################################################################################################
#########################################################################################################
Question: Write a function to calculate the total resistance of resistances in series in a given list
Source Python:
def cal_total_res_in_series(res_list : list)->float : 
     return sum(res_list )


Target Python:
def shift_and_scale(list_of_nums , std ) : 
     return [ ( x - mean ) / std for x in list_of_nums ] 
#########################################################################################################
#########################################################################################################
Question: Write a Python program to check number either positive , negative or zero
Source Python:
num = int ( input ( " Enter Integer Number : ") ) 
 if num = = 0 : 
 print ( " Zero Entered " ) 
 elif num > 0 : 
 print ( " Positive Number Entered " ) 
 elif num < 0 : 
 print ( " Negative Number Entered " )


Target Python:
import random 
 print random.sample(range(100 ) , 5 ) 
#########################################################################################################
#########################################################################################################
Question: Write a python program to print the uncommon elements in List
Source Python:

 test_list1 = [ [ 1 , 2 ] , [ 3 , 4 ] , [ 5 , 6 ] ] 
 test_list2 = [ [ 3 , 4 ] , [ 5 , 7 ] , [ 1 , 2 ] ] 

 res_list = [ ] 
 for i in test_list1 : 
     if i not in test_list2 : 
         res_list.append(i ) 
 for i in test_list2 : 
     if i not in test_list1 : 
         res_list.append(i ) 

 print ( " The uncommon of two lists is : " + str(res_list ) )


Target Python:
test_list = [ ( 1 , 5 ) , ( 1 ) , ( 3 ) , ( 1 ) , ( 3 ) ] 
 res = [ ] 
 for i , ( i ) for i , j in test_list : 
     for i ) ] )
#########################################################################################################
#########################################################################################################
Question: printing result
Source Python:
print("The filtered tuple : " + str(res ) )


Target Python:
print("The original dictionary is : " + str(test_dict ) ) 
#########################################################################################################
#########################################################################################################
Question: write a program to convert key - values list to flat dictionary
Source Python:
from itertools import product 
 test_dict = { ' month ' : [ 1 , 2 , 3 ] , 
              ' name ' : [ ' Jan ' , ' Feb ' , ' March ' ] } 

 print("The original dictionary is : " + str(test_dict ) ) 

 res = dict(zip(test_dict['month ' ] , test_dict['name ' ] ) ) 
 print("Flattened dictionary : " + str(res ) )


Target Python:
test_list = [ ( ' gfg ' , ' ) , ( ' 5 ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' ' '
#########################################################################################################
#########################################################################################################
Question: Write a function to return the volume of a hemi sphere
Source Python:
def cal_hemisphere_volume(radius : float)->float : 
     pi=3.14 
     return ( 2/3)*pi*(radius**3 )


Target Python:
def cal_cylinder_surf_area(height , radius ) : 
     pi=3.14 
     return 2*pi*radius**2*+2*pi*radius*height 
#########################################################################################################
#########################################################################################################
Question: write Python3 code to demonstrate working of   Sort tuple list by Nth element of tuple   using sort ( ) + lambda
Source Python:
test_list = [ ( 4 , 5 , 1 ) , ( 6 , 1 , 5 ) , ( 7 , 4 , 2 ) , ( 6 , 2 , 4 ) ] 
 print("The original list is : " + str(test_list ) ) 
 N = 1 
 test_list.sort(key = lambda x : x[N ] ) 
 print("List after sorting tuple by Nth index sort : " + str(test_list ) )


Target Python:
test_list = [ ( 1 , 5 ) , ( 1 ) , ( 3 ) , ( 1 ) , ( 3 ) ] 
 res = [ ( a , ( a , b ) for a , b ) for i , b , b ) in
#########################################################################################################
#########################################################################################################
Question: Write a Python function to check whether a person is eligible for voting or not based on their age
Source Python:
def vote_eligibility(age ) : 
	 if age>=18 : 
	     status="Eligible " 
	 else : 
	     status="Not Eligible " 
	 return status


Target Python:
def cal_circumference(r ) : 
     pi = 3.14 
     return 2*pi*r 
#########################################################################################################
#########################################################################################################
Question: Write a python program to swap tuple elements in list of tuples . Print the output .
Source Python:
test_list = [ ( 3 , 4 ) , ( 6 , 5 ) , ( 7 , 8) ] 

 res = [ ( sub[1 ] , sub[0 ] ) for sub in test_list ] 

 print("The swapped tuple list is : " + str(res ) )


Target Python:
test_list = [ ( 1 , 5 ) , ( 1 ) , ( 3 ) , ( 1 ) , ( 3 ) ] 
 res = [ ( a , ( a , b ) for i , b ) for i , b in test_list ] 

#########################################################################################################
#########################################################################################################
Question: Convert two lists into a dictionary
Source Python:
ItemId = [ 54 , 65 , 76 ] 
 names = [ " Hard Disk " , " Laptop " , " RAM " ] 
 itemDictionary = dict(zip(ItemId , names ) ) 
 print(itemDictionary )


Target Python:
test_list = [ ( 1 , 5 ) , ( 1 ) , ( 3 ) , ( 1 ) , ( 3 ) ] 
 res = [ ( a , ( a , b ) for i , b ) for i , b in test_list ] 

#########################################################################################################
#########################################################################################################
Question: Write a program that calculates and prints the value according to the given formula : Q = Square root of [ ( 2 * C * D)/H ]
Source Python:
import math 
 c=50 
 h=30 
 value = [ ] 
 items=[x for x in raw_input().split ( ' , ' ) ] 
 for d in items : 
     value.append(str(int(round(math.sqrt(2*c*float(d)/h ) ) ) ) ) 
 print ' , ' .join(value )


Target Python:
num = int(input("Enter a number : ") ) 
 if num > 0 : 
    print("Enter a positive number " ) 
 else : 
    # use while loop to iterate until zero 
    while(num > 0 ) : 
        sum + = num -= 1 
    while(num > 0
#########################################################################################################
#########################################################################################################
Question: create a tuple
Source Python:
tuplex = ( 2 , 4 , 3 , 5 , 4 , 6 , 7 , 8 , 6 , 1 )


Target Python:
test_list = [ ( 5 , 6 ) , ( 1 ) , ( 3 ) , ( 1 ) ] 
 print("The original list is : " + str(test_list ) 
 res = [ sub ) for ele in test_list : 
     for ele in sub : 
         res
#########################################################################################################
#########################################################################################################
Question: In[102 ] :
Source Python:
NA


Target Python:
NA 
#########################################################################################################
#########################################################################################################
Question: Counting total Upper Case in a string
Source Python:
str1 = " abc4234AFde " 
 digitCount = 0 
 for i in range(0,len(str1 ) ) : 
   char = str1[i ] 
   if(char.upper ( ) ) : 
     digitCount + = 1 
 print('Number total Upper Case : ' , digitCount )


Target Python:
str1 = " Hello ! It is a Good thing " 
 substr1 = " 
 substr2 = " 
 substr2 = " bad " 
 replaced_str = str1.replace(substr1 , substr2 ) 
 print("String after replace : " + str(replaced_str ) ) ) 
#########################################################################################################
#########################################################################################################
Question: Write a python function which wil return True if list parenthesis used in a input expression is valid , False otherwise
Source Python:
def isValid(s ) : 
     stack = [ ] 
     mapping = { ' ) ' : ' ( ' , ' } ' : ' { ' , ' ] ' : ' [ ' } 
     for char in s : 
         if char in mapping : 
             if not stack : 
                 return False 
             top = stack.pop ( ) 
             if mapping[char ] ! = top : 
                 return False 
         else : 
             stack.append(char ) 
     return not stack


Target Python:
def Fibonacci(n ) : 
     if n<0 : 
         print("Incorrect input " ) 
     elif n==1 : 
         print("Incorrect input " ) 
     elif n==1 : 
         return 1 
     elif n==1 : 
         return 1 
     else : 
         return Fibonacci(n-1)+Fibonacci(n-2 ) 
#########################################################################################################
#########################################################################################################
Question: The consequences of modifying a list when looping through it
Source Python:
a = [ 1 , 2 , 3 , 4 , 5 ] 
 for i in a : 
     if not i % 2 : 
         a.remove(i ) 
 print(a ) 
 b = [ 2 , 4 , 5 , 6 ] 
 for i in b : 
      if not i % 2 : 
          b.remove(i ) 
 print(b )


Target Python:
for i in range(len(X ) ) : 
    # iterate through columns of Y 
    for j in range(len(Y[0 ] ) : 
        for k in range(len(Y ) : 
            result[i][j ] + = X[i][k ] * Y[k][j ] 
 for r in result : 
    print(r ) 
#########################################################################################################

Question: Write a function to remove punctuation from the string
Source Python:
def r_punc ( ) : 
     test_str = " end , is best : for ! Nlp ; " 
     print("The original string is : " + test_str ) 
     punc = r'!()-[]{};:\ , < > ./?@#$%^&*_~ ' 
     for ele in test_str : 
         if ele in punc : 
             test_str = test_str.replace(ele , " ") 
     print("The string after punctuation filter : " + test_str )


Target Python:
def check(string ) : 
     s = p = { ' 0 ' } 
     if s = = { ' } or p = { ' } : 
         return True 
     else : 
         return False 
```



### Model Output with min_freq = 1

```
Question: 47 write a python function that accepts a valid path and changes the current working directory
Source Python:
import os 
 def change_dir(path ) : 
     return os.chdir(path )


Target Python:
def read_and_print_file(filepath ) : 
     with open(filepath , " r " ) as infile : 
         print ( infile.read ( ) ) 
#########################################################################################################
#########################################################################################################
Question: Write a Python program to return a set of all elements in either A or B , but not both
Source Python:
set1 = { 10 , 20 , 30 , 40 , 50 } 
 set2 = { 30 , 40 , 50 , 60 , 70 } 
 print(set1.symmetric_difference(set2 ) )


Target Python:
a = 60 
 b = 13 
 c = a python language 
 while a < 0 : 
   print("True " ) 
   else : 
     if(n%i==0 ) 
 print("First Set after removing common element " , largest ) 
#########################################################################################################
#########################################################################################################
Question: write a python function to return an iterator over the last n items
Source Python:
def tail(n , iterable ) : 
     from collections import deque 
     return iter(deque(iterable , maxlen = n ) )


Target Python:
def digits_sum ( ) : 
	 n = 2**20 
	 ans = sum(int(c ) for c in str(n ) ) 
	 return str(ans ) 
#########################################################################################################
#########################################################################################################
Question: write a python function to convert a string   into xml
Source Python:
import xml.etree . ElementTree as ET 
 root = ET.fromstring(country_data_as_string )


Target Python:
import re 
 def check(email ) : 
     regex = ' ^[a - z0 - 9]+[\._]?[a - z0 - 9]+[@]\w+[.]\w{2,3}$ ' 
     if(re.search(regex , email ) ) : 
         print("Valid Email " ) 
     else : 
         print("Invalid Email " ) 
#########################################################################################################
#########################################################################################################
Question: Write a function that returns runs a garbage collector
Source Python:
def clear_memory ( ) : 
     import gc 
     gc.collect ( )


Target Python:
def to_upper(s ) : 
     return s.upper ( ) 
#########################################################################################################
#########################################################################################################
Question: Write python function to generate valid parenthesis , number of parenthesis is given as input
Source Python:
def generateParenthesis(n ) : 

     def backtrack(S= ' ' , left=0 , right=0 ) : 
         if len(S ) = = 2*n : 
             output.append(S ) 
             return 
         if left < n : 
             backtrack(S+ ' ( ' , left+1 , right ) 
         if right < left : 
             backtrack(S+ ' ) ' , left , right+1 ) 

     output = [ ] 
     backtrack ( ) 
     return output


Target Python:
def power(base , exp ) : 
     if(exp==1 ) : 
         return(base ) 
     if(exp!=1 ) : 
         return(base*power(base , exp-1 ) ) 
#########################################################################################################
#########################################################################################################
Question: Write a python program to implement bubble sort and print the result
Source Python:
from random import randint 
 N = 7 
 a = [ ] 
 for i in range(N ) : 
     a.append(randint(1 , 20 ) ) 
 print(a ) 
 for i in range(N-1 ) : 
     for j in range(N - i-1 ) : 
         if a[j ] > a[j+1 ] : 
             b = a[j ] 
             a[j ] = a[j+1 ] 
             a[j+1 ] = b 
 print(a )


Target Python:
def stoogesort(arr , l , h ) : 
     if l > = h : 
         return 
     if arr[l ] > arr[h ] : 
         arr[l ] = arr[h ] = arr[h ] 
         t = arr[l ] = arr[h ] = arr[h ] 
         t 
     if h
#########################################################################################################
#########################################################################################################
Question: Write a Python Program to Sort the List According to the Second Element in Sublist
Source Python:
a=[['A',34],['B',21],['C',26 ] ] 
 for i in range(0,len(a ) ) : 
     for j in range(0,len(a)-i-1 ) : 
         if(a[j][1]>a[j+1][1 ] ) : 
             temp = a[j ] 
             a[j]=a[j+1 ] 
             a[j+1]=temp


Target Python:
NA 
#########################################################################################################
#########################################################################################################
Question: Write a function to return the real of the roots of a quadratic equation else return None ax**2 + bx + c = 0
Source Python:
def roots_of_qad_eq(a : float , b : float , c : float ) : 
     d = b**2 - 4*a*c 
     if d > = 0 : 
         return ( -b+(d)**(1/2))/2*a,(-b-(d)**(1/2))/2*a 
     else : 
         return None


Target Python:
def sum_of_roots(a : float , c : float ) : 
     if a : 
         return c / a 
     else : 
         return None 
#########################################################################################################
#########################################################################################################
Question: Write a function to calculate the Temprature T of ideal gas based on ideal gas equation Pressure P and Volume V given
Source Python:
def find_temp_of_ideal_gas(pressure : float , volume : float , n : float)->float : 
     r = 8.3145 # gas constant R 
     return ( pressure*volume)/n*r


Target Python:
def get_ci(p : float , r : float , t : float)->float : 
     return round(p*((1+(r/(n*100)))**(n*t ) ) - p,2 ) 
#########################################################################################################
#########################################################################################################
Question: Define a function that can accept two strings as input and print the string with maximum length in console . If two strings have the same length , then the function should print al l strings line by line .
Source Python:
def printValue(s1,s2 ) : 
	 len1 = len(s1 ) 
	 len2 = len(s2 ) 
	 if len1 > len2 : 
		 print s1 
	 elif len2 > len1 : 
		 print s2 
	 else : 
		 print s1 
		 print s2


Target Python:
def rotate_right(input , d ) : 

     Rfirst = input[0 : len(input)-d ] 
     Rsecond = input[len(input)-d : ] 
     return ( Rsecond + Rfirst ) 
#########################################################################################################
#########################################################################################################
Question: write a python function that would return the sum of first n natural numbers , where n is the input
Source Python:
def sum_first_n(n ) : 
     return ( n * ( n+1 ) ) // 2


Target Python:
def sum_first_n_recursive(n ) : 
     if n = = 0 : 
         return 0 
     return 0 
     return sum_first_n_recursive(n-1 ) + n 
#########################################################################################################
#########################################################################################################
Question: how to check if a list is a subset of another list
Source Python:
if(all(x in test_list for x in sub_list ) ) : 
     flag = True


Target Python:
l = [ ] 
 if not l : 
 print("List is empty " ) 
#########################################################################################################
#########################################################################################################
Question: Write a Python program to check / test multiple variables against a value
Source Python:
a = 10 
 b = 20 
 c = 30 
 if 10 in { a , b , c } : 
   print("True " ) 
 else : 
   print("False " )


Target Python:
a = 10 
 b = 20 
 c = 30 
 if ( c < 30 ) : 
   print("True " ) 
 elif ( ) 
 else : 
 else : 
 break 
 print("The original dictionary is : " + str(sample_dict ) 
 res = { }
#########################################################################################################
#########################################################################################################
Question: how to add element at first position in array python
Source Python:
x = [ 1,3,4 ] 
 a = 2 
 x.insert(1,a )


Target Python:
my_list = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 8 , 10 ] 
 print(my_list[:5 ] ) 
#########################################################################################################
#########################################################################################################
Question: Write a function that returns the sum of digits of a given number
Source Python:
def digisum(num ) : 
     sum_=0 
     while num > 0 : 
         dig = num % 10 
         sum_+=dig 
         num//=10 
     return sum _


Target Python:
def sum_of_digits(num ) : 
   if num = = 0 : 
     return num % 10 
   else : 
     return num % 10 + sum_of_digits(int(num / 10 ) ) 
#########################################################################################################
#########################################################################################################
Question: Counting total Digits in a string
Source Python:
str1 = " abc4234AFde " 
 digitCount = 0 
 for i in range(0,len(str1 ) ) : 
   char = str1[i ] 
   if(char.isdigit ( ) ) : 
     digitCount + = 1 
 print('Number of digits : ' , digitCount )


Target Python:
str1 = " abc4234AFde " 
 digitCount = 0 
 for i in range(0,len(str1 ) ) : 
   char = str1[i ] 
   if(char.isalpha ( ) ) : 
     digitCount + = 1 
 print('Number of digits : ' , digitCount ) 
#########################################################################################################
#########################################################################################################
Question: sorted ( ) to sort , lambda provides key - value addition
Source Python:
res = sorted(test_dict.items ( ) , key = lambda sub : sub[0 ] + sub[1 ] )


Target Python:
test_dict = { ' gfg ' : 5 , ' is ' : 3 , ' best ' : 4 } 
#########################################################################################################
#########################################################################################################
Question: write a function to compress a given string . Suppose a character ' c ' occurs consecutively X times in the string . Replace these consecutive occurrences of the character ' c ' with   ( X , c ) in the string .
Source Python:
def compress(text ) : 
     from itertools import groupby 
     for k , g in groupby(text ) : 
         print ( " ( { } , { } ) " .format(len(list(g ) ) , k ) , end= " ")


Target Python:
def moveSpaces(str1 ) : 
     str1 = str1 = " 
     str1 = " returns the str1 
     if char in str1 : 
         str1 = = ' I ' 
     return str1 

 def inner 
#########################################################################################################
#########################################################################################################
Question: 3x4 matrix
Source Python:
Y = [ [ 5,8,1,2 ] , 
     [ 6,7,3,0 ] , 
     [ 4,5,9,1 ] ]


Target Python:
Y = [ [ 5,8,1,2 ] , 
     [ 6,7,3,0 ] , 
     [ 4,5,9,1 ] ] 
#########################################################################################################
#########################################################################################################
Question: write a program to print the binary value of the numbers from 1 to N
Source Python:
n = int(input("Enter the value of N : ") ) 
 for i in range(1 , n+1 ) : 
     print("Binary value of " , i , " is : " , bin(i ) )


Target Python:
num = 16 
 if num < 0 : 
    print("Enter a positive number " ) 
 else : 
    sum = 0 
    while(num > 0 ) : 
        sum + = num 
        num -= 1 
    print("The sum is " , sum ) 
#########################################################################################################
#########################################################################################################
Question: Write a function to return the total surface area of a cube of side a
Source Python:
def cal_surface_area_cube(a ) : 
     return 6*(a**2 )


Target Python:
def cal_cylinder_surf_area(height , radius ) : 
     pi=3.14 
     return 2*pi*radius**2*+2*pi*radius*height 
#########################################################################################################
#########################################################################################################
Question: Write a Python program to reverse a tuple .
Source Python:
NA


Target Python:
Tuple = ( 10,20 ) 
 def sizeOfTuple(tup ) : 
   return f'Size of Tuple : { str(Tuple.__sizeof _ ( ) ) } bytes ' 
#########################################################################################################
#########################################################################################################
Question: write a python function to return the square root of a number
Source Python:
def get_sqrt(i ) : 
     import math 
     return(math.sqrt(i ) )


Target Python:
def square(x ) : 
     return x**2 
#########################################################################################################
#########################################################################################################
Question: Write a Python function to the push the first number to the end of a list .
Source Python:
def move_last(num_list ) : 
     a = [ num_list[0 ] for i in range(num_list.count(num_list[0 ] ) ) ] 
     x = [ i for i in num_list if i ! = num_list[0 ] ] 
     x.extend(a ) 
     return(x )


Target Python:
def move_last(num_list ) : 
     a = [ num_list[0 ] for i in range(num_list.count(num_list[0 ] ) ) ) ] 
     x = [ i for i in l ] 
     x in l : 
         x + = [ i for j in l ] 
     return [ i for
#########################################################################################################
#########################################################################################################
```



### Model Output with min_freq = 2

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

