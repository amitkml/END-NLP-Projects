[TOC]

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

### Model with TEXT.build_vocab(train_data, min_freq = 1)

Have kept min_freq to 1 and this has helped me to get rid of <unk>.

Model output has been kept as default one with 3 Encoder and Decoder layer.

```
INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = len(TEXT.vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
```

**Model performance on Test data has been quite good and best among all my experimentation.**

```
| Test Loss: 2.119 | Test PPL:   8.325 |
```

Model Output has been quite good and quite a lot areas where model has worked quite well.

```
Question: Use a list comprehension to square each odd number in a list . The list is input by a sequence of comma - separated numbers .
Source Python:
values = input ( ) 
 numbers = [ x for x in values.split ( " , " ) if int(x)%2!=0 ] 
 print(",".join(numbers ) )


Target Python:
list1 = [ 1 , 2 , 3 , 4 , 5 ] 
 list2 = [ 5 , 6 , 7 , 2 , 3 , 8 ] 
 final = [ a+b for a in list1 for b in list1 if a ! = count+1 ] 
 print("Odd
#########################################################################################################
#########################################################################################################
Question: write a function to check if a lower case letter exists in a given string
Source Python:
def check_lower(str1 ) : 

     for char in str1 : 
         k = char.islower ( ) 
         if k = = True : 
             return True 
     if(k ! = 1 ) : 
         return False


Target Python:
def check_upper(str1 ) : 

     for char in str1 : 
         k = char.isupper ( ) 
         if k = = True : 
             return True 
     if(k ! = 1 ) : 
         return False 
#########################################################################################################
#########################################################################################################
Question: access Last characters in a string
Source Python:
word = " Hello World " 
 letter = word[-1 ] 
 print(f"First Charecter in String:{letter } " )


Target Python:
word = " Hello World " 
 check = word.isalpha ( ) 
 print(f"All char are alphabetic?:{check } " ) 
#########################################################################################################
#########################################################################################################
Question: Python Program to Make a Simple Calculator
Source Python:
NA


Target Python:
NA 
#########################################################################################################
#########################################################################################################
Question: Write a function to return the area of a trapezium with base a base b and height h between parallel sides
Source Python:
def cal_area_trapezium(a , b , h ) : 
     return h*(a+b)/2


Target Python:
def cal_area_trapezium(a , b , h ) : 
     return h*(a+b)/2 
#########################################################################################################
#########################################################################################################
Question: Write a python function that prints the Contents of a File in Reverse Order
Source Python:
def reverse_content(filename ) : 
     for line in reversed(list(open(filename ) ) ) : 
         print(line.rstrip ( ) )


Target Python:
def read_and_print_file(filepath ) : 
     with open(filepath , " r " ) as infile : 
         print ( infile.read ( ) ) 
#########################################################################################################
#########################################################################################################
Question: write a function to remove i - th indexed character in a given string
Source Python:
def remove_char(string , i ) : 
     str1 = string [ : i ] 
     str2 = string[i + 1 : ] 

     return str1 + str2


Target Python:
def remove(string , i ) : 
     i = i + i 
     return i 
#########################################################################################################
#########################################################################################################
Question: Write a Python Program to Multiply All the Items in a Dictionary and print the result
Source Python:
d={'A':10,'B':10,'C':239 } 
 tot=1 
 for i in d : 
     tot = tot*d[i ] 
 print(tot )


Target Python:
dict1 = { ' a ' : 10 , ' b ' : 10 } 
 dict2 = { ' : 300 , ' c ' : 300 } 
 for key in dict1 : 
     if key in dict2 : 
         dict2[key ] = dict2[key ] = dict2[key ]
#########################################################################################################
#########################################################################################################
Question: Write a Python function to return woodall numbers
Source Python:
NA


Target Python:
def is_prod_even(num1 , num2 ) : 
    prod = num1 * num2 
    return not prod % 2 
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
Question: Python Challenges : Check a sequence of numbers is a geometric progression or not
Source Python:
def is_geometric(li ) : 
     if len(li ) < = 1 : 
         return True 
     # Calculate ratio 
     ratio = li[1]/float(li[0 ] ) 
     # Check the ratio of the remaining 
     for i in range(1 , len(li ) ) : 
         if li[i]/float(li[i-1 ] ) ! = ratio : 
             return False 
     return True


Target Python:
def is_prod_even(num1 , num2 ) : 
    sum = num1 + num2 
    return not sum not sum % 2 
#########################################################################################################
#########################################################################################################
Question: write a function to replace all occurances of a substring in a string
Source Python:
str1 = " Hello ! It is a Good thing " 
 substr1 = " Good " 
 substr2 = " bad " 
 replaced_str = str1.replace(substr1 , substr2 ) 
 print("String after replace : " + str(replaced_str ) )


Target Python:
str1 = " It is wonderful and sunny day for a picnic in the park " 
 str_len = 5 
 res_str = [ ] 

 text = str1.split ( " ") 

 for x in text : 
     if len(x ) < str_len : 
         res_str.append(x ) 
 print("Words
#########################################################################################################
#########################################################################################################
Question: write a python function to check if a given string is a palindrome
Source Python:
def is_palindrome(string ) : 
    return string = = string[::-1 ]


Target Python:
def isPalindrome(s ) : 
     return s = = s[::-1 ] 
#########################################################################################################
#########################################################################################################
Question: Write a Python function to calculate the geometric sum of n-1 .
Source Python:
def geometric_sum(n ) : 
   if n < 0 : 
     return 0 
   else : 
     return 1 / ( pow(2 , n ) ) + geometric_sum(n - 1 )


Target Python:
def factorial(n ) : 
     if n = = 0 : 
         return 1 
     else : 
         return n*factorial(n-1 ) 
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
def bmi_calculator(height , weight ) : 
	 bmi = weight/(height**2 ) 
	 return bmi 
#########################################################################################################
#########################################################################################################
Question: use built - in function filter to filter empty value
Source Python:
new_str_list = list(filter(None , str_list ) ) 
 print("After removing empty strings " ) 
 print(new_str_list )


Target Python:
test_list = [ { ' gfg ' : [ 5 , 6 ] , 
              ' best ' : [ 7 ] , 
              ' : [ 10 ] , 
              ' best ' : [ 10 ] , 
              ' : [ 10 ] , 
              ' CS '
#########################################################################################################
#########################################################################################################
Question: 3 write a python program to convert a string to a char array
Source Python:
def char_array(string ) : 
     return list(string )


Target Python:
word = " Hello World " 
 check = word.isalpha ( ) 
 print(f"All char are alphabetic?:{check } " ) 
#########################################################################################################
#########################################################################################################
Question: Write a function to calculate volume of Triangular Pyramid
Source Python:
def volumeTriangular(a , b , h ) : 
     return ( 0.1666 ) * a * b * h


Target Python:
def dot_product(a , b ) : 
     return sum ( e[0]*e[1 ] for e in zip(a , b ) ) 
#########################################################################################################
#########################################################################################################
Question: Write a function to find power of number using recursion
Source Python:
def power(N , P ) : 
     if ( P = = 0 or P = = 1 ) : 
         return N 
     else : 
         return ( N * power(N , P - 1 ) ) 
 print(power(5 , 2 ) )


Target Python:
def power(N , P ) : 
     if ( P = = 0 and ( P = 1 ) : 
         return N 
     else : 
         return ( N * power(N , P - 1 ) 
 print(power(5 , 2 ) ) 
#########################################################################################################
#########################################################################################################
Question: write a program to print perfect numbers from the given list of integers
Source Python:
def checkPerfectNum(n ) : 
	 i = 2;sum = 1 ; 
	 while(i < = n//2 ) : 
		 if ( n % i = = 0 ) : 
			 sum + = i 

		 i + = 1 
		 if sum = = n : 
			 print(n , end= ' ' ) 
 if _ _ name _ _ = = " _ _ main _ _ " : 
	 print("Enter list of integers : ") 
	 list_of_intgers = list(map(int , input().split ( ) ) ) 
	 print("Given list of integers:",list_of_intgers ) 
	 print("Perfect numbers present in the list is : ") 
	 for num in list_of_intgers : 
		 checkPerfectNum(num )


Target Python:
a = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 8 , 9 , 10 ] 
 b = [ 2 , 8 , 9 , 10 , 15 ] 
 for i in zip(a , b ) : 
     print(matrix[i][1
#########################################################################################################
#########################################################################################################
Question: 3x3 matrix
Source Python:
X = [ [ 12,7,3 ] , 
     [ 4 , 5,6 ] , 
     [ 7 , 8,9 ] ]


Target Python:
X = [ [ 12,7,3 ] , 
     [ 4 , 5,6 ] , 
     [ 7 , 8,9 ] ] 
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
list1 = [ 1 , 2 , 3 , 4 , 5 ] 
 list2 = [ 5 , 6 , 7 , 8 , 10 ] 
 final = [ a , b for a in list1 if b not in list1 ] 
 print("New list after removing all
#########################################################################################################
#########################################################################################################
Question: Write a function to return the median of numbers in a list
Source Python:
def cal_median(num_list : list)->float : 
     if num_list : 
         if len(num_list)%2 ! = 0 : 
             return sorted(num_list)[int(len(num_list)/2 ) - 1 ] 
         else : 
             return ( sorted(num_list)[int(len(num_list)/2 ) - 1 ] + sorted(num_list)[int(len(num_list)/2)])/2 
     else : 
         return None


Target Python:
def cal_median(num_list : list)->float : 
     if num_list : 
         if len(num_list)%2 ! = 0 : 
             return sorted(num_list)[int(len(num_list)/2 ) - 1 ] - 1 ] - 1 ] + sorted(num_list)[int(len(num_list)/2)])/2 
     else : 
         return None 
#########################################################################################################
#########################################################################################################
Question: Write a function to identify to count no of instances of a value   inside a dictionary
Source Python:
def count_value(d : dict , value)->bool : 
     return list(v = = value for v in dict.values()).count(True )


Target Python:
def invert_dict_non_unique(my_dict ) : 
   my_inverted_dict = dict ( ) 
   for key , value in my_dict.items ( ) : 
       my_inverted_dict.setdefault(value , list()).append(key ) 
   return my_inverted_dict 
#########################################################################################################
#########################################################################################################
Question: Python String Operations
Source Python:
str1 = ' Good ' 
 str2 = ' Morning ! '


Target Python:
str1 = " It is a great day " 
 print("The string are : " , str1 ) 
 res = str1.split ( " + str(res ) ) 
```



### Model with TEXT.build_vocab(train_data, min_freq = 2)

### Model with TEXT.build_vocab(train_data, min_freq = 1) and custom tokenizer to handle python special characters

### Model with TEXT.build_vocab(train_data, min_freq = 1) and higher size of hidden layer

Model Encoder and Decoder dimension has been increased from 512 to 1024. Model File name loaded in [Model_Experiment_6](transformer-based-model-python-code-generator/src/END_NLP_CAPSTONE_PROJECT_English_Python_Code_Transformer_6_0.ipynb)

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



