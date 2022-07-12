

# Julia

> What is julia?
Julia is a high level programming language, like python or R, made
specifically for high performance numerical computing
> Why would you not use python?
Python is slow, OOP is hard, having to do vectorized numpy code is hard and
hard to read. When python needs to be fast, you also need to know C++ or
something like that. You could use numba or something else JIT compiled as
well, but that sucks!

# Why julia

Julia gives you the high level, dynamic, interpreted language you want and
need to be happy, but has the speed of a low level compiled language like C,
C++, and Rust. See [The benchmarks](https://julialang.org/benchmarks/)!

# Why is python slow?

```whiteboard 
parse               bytecode
      ┌──────────┐       ┌───────────┐       ┌───────┐
      │my_file.py├──────►│my_file.pyc│──────►│opcodes│
      └──────────┘       └───────────┘       └───┬───┘
                                                 │
                                                 ▼
                                           store in memory
   ┌──────────────────┐  ┌────────────┐          │
   │call compiled code│◄─┤call opcodes│ ◄────────┘
   └─────────┬────────┘  └────────────┘
             └───────┐
                     ▼
           ╔════════════════════╗
           ║MACHINE INSTRUCTIONS║
           ╚════════════════════╝
```
Code is parsed (which is very slow as well, python function lookup is slow),
interpreter turns it into bytecode, then python VM turns bytecode into opcodes
which are stored in memory, which are then called by the cpu which load
compiled code up, and then the cpu instructions in the compiled code are
executed by your machine. This is a lot of steps!
Numpy is fast because you are closer to the compiled code, instead of calling
the same bit of compiled code 1000 times in a for loop, someone wrote that for
loop in fortran or C, so you need to go through this crazy process less!


# How does julia work?

1. **Parsing**: Code is parsed as an abstract syntax tree (AST)
2. **Expansion**: Any macros or metaprograms are expanded, yielding a different
   AST (we will touch on this further) 
3. **Lowering**: Code is transformed into simpler code by julia
4. **Typing**: Possible types are inferred by the julia interpreter. This is
   important, knowing what type the inputs are means that the compiler can pick
   a more optimized function 
5. **Translation**: Simplified, typed AST is translated into LLVM (a compiler)
   instructions
6. **Compilation**: LLVM transforms code directly into machine instructions
This means that your julia code doesnt have to call up a bunch of other code
to run, it is actually directly turned into the code, but this is done
dynamically, so it is still an interpreted language, and the
interperter/compiler figure out the types for you, so you dont need to have
some horrible static typed code (look at C code or Java code)
The best of both worlds! This means that a for loop is just as fast if not 

# Is julia hard?

- Can you write and use a function?
- Can you do a for loop/map/while
- Can you do sets and lists and vectors and dicts?
- Can you use a dataframe?


# Other reasons to use julia: functional programming

Python is an OOP language, meaning everything is an object! What does this
mean?

```python

>>> type(1)
>>> int
>>> int.__add__

```


Julia is a functional language and uses a paradigm called _multiple dispatch_.
Lets look at the same example:

```julia 

typeof(1)
@which 1+2
methods(+)
@which 4.5 + 1.
@which 1+4.5

```

Something fishy is going on here!

# No classes

Thats right! There are no classes! But this is not the end of the world! We
can do anything you do in python, except differently! Instead of building
classes for complicated datatypes, we just extend functions for them!

## Simple Demo

```julia 

    # code is still code
    x = 4
    y = 5
    x + y
    
    
    # functions are still functions
    function demo(a, b, c = 2)
        return (a + b) ^ c
    end
    # inline functions!
    f(x) = x + 2
    # nameless functions!
        (x -> x + 2)(2)
    
    # calling
    demo(4, 5, 6)
    
    # map
    a = map((x -> demo(x, 5, 6)), 1:5)
    
    # broadcasting! magic!
    b = demo.(1:5, 5, 6)
    
    # for loops!
    c = Int64[]
    for x in 1:5 
        push!(c, demo(x, 5, 6))
    end
    
    a == b == c
    
    # talk about lists for a moment
    
    # dicts!
    d = Dict(zip(["a", "b", "c"], [1, 2, 3])) # pythonic
    d["a"]
    
    # symbols!
    d = Dict(:a => 1, :b => 2, :c => 3)
    d[:a]

    # sets
    Set(repeat(collect(1:5), 55))

```
It is just fancy python

## Getting fancy: python code

```python

class Point(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        def __add__(self, p):
            if isinstance(p, Point):
                return Point(self.a + p.a, self.b + p.b)
            elif isinstance(p, (float, int)):
                return Point(self.a + p, self.b + p)
            else:
                raise TypeError(f"Cannot add {type(p)} to a Point!")

    >>> p = Point(1, 2)
    >>> q = Point(3, 4)
    >>> w = Point(1.2, 3.5)
    >>> z = p + q + 14
    >>> z

    class Line(object):
        def __init__(self, p1, p2, p3):
            # we could subclass, but this means everything that works on a point
            # must work on a line
            self.p1 = p1
            self.p2 = p2
            self.p3 = p3
        def __add__(self, l):
            return Line(self.p1 + l.p1, self.p2 + l.p2, self.p3 + l.p3)

    >>> L1 = Line(1, 2, 3)
    >>> L2 = Line(4, 5, 6)
    >>> L1 + L2
    >>> L3 = Line(p, q, z)
    >>> L4 = Line(q, z, p)
    >>> L3 + L4

```

## Julia style

```julia
import Base:+
       struct Point{T}
           x::T
           y::T
       end

   p = Point(1, 2)
   q = Point(3, 4)
   w = Point(1.2, 3.5)

   (+)(a::Point, b::Point) = Point(a.x + b.x, a.y + b.y)
   # without a bunch of if statements, this is not possible in python
   (+)(a::Point, b::Number) = Point(a.x + b, a.y + b)

   z = p + q 
   pp = Point(p, q)
   qq = Point(q, p)
   zz = pp + qq
   z_shifted = zz + 4.5

   struct Line{T}
       p1::T
       p2::T
       p3::T
   end

   (+)(a::Line, b::Line) = Line(a.p1 + b.p1, a.p2 + b.p2, a.p3 + b.p3)
   (+)(a::Line, b::Number) = Line(a.p1 + b, a.p2 + b, a.p3 + b)
   L1 = Line(1, 2, 3)
   L2 = Line(4, 5, 6)
   L1 + L2 + 23
   L3 = Line(p, q, z)
   L4 = Line(q, z, p)
   L3 + L4 + 467.234
```

Which is clearer?

# Digging deeper

Making a DSL! (you really can't do this in python). Follows
[this post](https://julialang.org/blog/2017/08/dsl/)

a DSL is a small, simple language which is optimized for a specific task
domain specific language (DSL) 

## Making a simple DSL:

Suppose we want to create some sort of `NaiveModel` type, which runs whatever
function we put into it on whatever else is passed to it. We could do this
pretty easily:
```julia 

    struct NaiveModel{F}
        f::F 
    end

    m = NaiveModel(x -> 2x)
    m.f(23)

    # overload the function call syntax
    (m::NaiveModel)(x) = m.f(x)
    m(23)

```
Pretty cool! But so far we have just created a function!

## Brief tangent: macros

A macro is a function that takes code and returns code! Pretty neat!

### Code as data

```julia 

    ex = :(x -> 2x)
    ex
    typeof(ex)

```

### Manipulating code

```julia 

    function make_function(ex::Expr)
        return :(x -> $ex)
    end
    ex = :(2x)
    make_function(ex)

    function make_model(ex::Expr)
        return :(NaiveModel($ex))
    end

    make_model(make_function(ex))
    m = eval(make_model(make_function(:(2x))))
    m(23)

```
This is ugly and painful!

### Macros!

```julia 

    macro naive_model(ex)
        @show ex
        @show typeof(ex)
        return nothing
    end

    # does all the hard work for us!
    m = @naive_model 2x 

    macro naive_model(ex)
        return make_model(make_function(ex))
    end

    m = @naive_model 2x

    m(23)
    m(44)

```

# Real world example: JuMP.jl

follows [Sudoko example](https://jump.dev/JuMP.jl/stable/tutorials/linear/sudoku/)
Suppose we want to solve sudoku. We have a problem with the following
constraints:
- The numbers 1 to 9 must appear in each 3x3 square
- The numbers 1 to 9 must appear in each row
- The numbers 1 to 9 must appear in each column
and the following board
```julia 
[
        5 3 0 │ 0 7 0 │ 0 0 0
        6 0 0 │ 1 9 5 │ 0 0 0
        0 9 8 │ 0 0 0 │ 0 6 0
        ──────┼───────┼──────
        8 0 0 │ 0 6 0 │ 0 0 3
        4 0 0 │ 8 0 3 │ 0 0 1
        7 0 0 │ 0 2 0 │ 0 0 6
        ──────┼───────┼──────
        0 6 0 │ 0 0 0 │ 2 8 0
        0 0 0 │ 4 1 9 │ 0 0 5
        0 0 0 │ 0 8 0 │ 0 7 9
    ]                     
   
```
We can solve this using a very similar DSL, `JuMP.jl` 
```julia 
using JuMP
  using HiGHS # optimizer library

  sudoku = Model(HiGHS.Optimizer)
```
Now, lets set up our problem as follows:

With this dataset definition, we can add variables to our model, using the JuMP DSL
```julia 
@variable(sudoku, x[i = 1:9, j = 1:9, k = 1:9], Bin)
  sudoku
```
Next, we can constrain our optimization problem! The first constraint is very
lame: there can only be one number per sudoku cell:
```julia 

  for i in 1:9  ## For each row
      for j in 1:9  ## and each column
          # Sum across all the possible digits. One and only one of the digits
          # can be in this cell, so the sum must be equal to one.
          @constraint(sudoku, sum(x[i, j, k] for k in 1:9) == 1)
      end
  end

```
Next, we need to add the constraint that every number in each row or column is
unique:
```julia 

  for ind in 1:9
      for k in 1:9
            # rows have unique numbers
            @constraint(sudoku, sum(x[ind, j, k] for j in 1:9) == 1)
            # columns have unique numbers
            @constraint(sudoku, sum(x[i, ind, k] for i in 1:9) == 1)
      end
  end
   
```
Next the tricky one, we need to say that each 3x3 grid can have 1-9. We can
accomplish this by starting at the top left corner, and then indexing 2 to the
right and 2 down. Brief syntax trick: comprehensions!
```julia 

  [x for x in 1:3:7]
  [[x for x in 1:3:7] for y in 1:3:7]
  [[[r, c] for r in i:(i+2), c in j:(j+2)] for i in 1:3:7 for j in 1:3:7]
   
```
```julia
for i in 1:3:7
      for j in 1:3:7
          for k in 1:9
              @constraint(
                  sudoku,
                  sum(x[r, c, k] for r in i:(i+2), c in j:(j+2)) == 1
              )
          end
      end
  end
   
```
Now lets solve sudoku:
```julia 
board = [
     5 3 0 0 7 0 0 0 0
     6 0 0 1 9 5 0 0 0
     0 9 8 0 0 0 0 6 0
     8 0 0 0 6 0 0 0 3
     4 0 0 8 0 3 0 0 1
     7 0 0 0 2 0 0 0 6
     0 6 0 0 0 0 2 8 0
     0 0 0 4 1 9 0 0 5
     0 0 0 0 8 0 0 7 9
  ]

  for i in 1:9
      for j in 1:9
          if board[i,j] != 0
              # some magic here because its a dsl
              fix(x[i, j, board[i,j]], 1; force = true)
          end
      end
  end

  optimize!(sudoku)

  x_val = value.(x)
  sol = zeros(Int, 9, 9)  # 9x9 matrix of integers
  for i in 1:9
      for j in 1:9
          for k in 1:9
              # Integer programs are solved as a series of linear programs so the
              # values might not be precisely 0 and 1. We can round them to
              # the nearest integer to make it easier.
              if round(Int, x_val[i, j, k]) == 1
                  sol[i, j] = k
              end
          end
      end
  end

  sol
```
