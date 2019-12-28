
x1 = 2.0
x2 = 3.0

ReLu = lambda x: max(0.0, x)
ReLuDer = lambda x: 1 if x > 0 else 0
error_fn = lambda prediction, target: 0.5 * (target - prediction) ** 2

# input
a1 = x1
a2 = x2

w11 = 0.11
w12 = 0.21

w21 = 0.12
w22 = 0.08

w1o = 0.14
w2o = 0.15

y = 1
n = 0.5

# foward 
# layer 1
zh1 = (w11 * a1) + (w12 * a2)
zh2 = (w21 * a1) + (w22 * a2)
#print(f"zh1 = {zh1}")
#print(f"zh2 = {zh2}")

h0 = 1
h1 = ReLu(zh1)
h2 = ReLu(zh2)
#print(f"h1 = {h1}")
#print(f"h2 = {h2}")

# layer 2
zo1 = (w1o * h1) + (w2o * h2)
o1 = ReLu(zo1)
error = error_fn(o1, y)

#print(f"zo1 = {zo1}")
print(f"o1 = {o1}")
print(f"error = {error}")

# Back

# Last layer
d_Etotal_d_out = (o1 - y)
#print(f"d_Etotal_d_out = {d_Etotal_d_out}")

d_out_d_zo1 = ReLuDer(o1)
#print(f"d_out_d_zo1 = {d_out_d_zo1}")

d_zo1_d_w1o = h1
#print(f"d_zo1_d_w1o = {d_zo1_d_w1o}")

d_zo1_d_w2o = h2
#print(f"d_zo1_d_w2o = {d_zo1_d_w2o}")

d_Etotal_d_w1o = d_Etotal_d_out * d_out_d_zo1 * d_zo1_d_w1o
#print(f"d_Etotal_d_w1o = {d_Etotal_d_w1o}")

d_Etotal_d_w2o = d_Etotal_d_out * d_out_d_zo1 * d_zo1_d_w2o
#print(f"d_Etotal_d_w1o = {d_Etotal_d_w2o}")


# Previous layer
d_w1o_d_h1 = w1o
d_h1_d_zh1 = 1
d_zh1_d_w11 = a1

d_Etotal_d_w11 = d_Etotal_d_w1o * d_w1o_d_h1 * d_h1_d_zh1 * d_zh1_d_w11
#print(f"d_Etotal_d_w11 = {d_Etotal_d_w11}")

d_w1o_d_h1 = w1o
d_h1_d_zh1 = 1
d_zh1_d_w12 = a2

d_Etotal_d_w12 = d_Etotal_d_w1o * d_w1o_d_h1 * d_h1_d_zh1 * d_zh1_d_w12
#print(f"d_Etotal_d_w11 = {d_Etotal_d_w12}")

d_w2o_d_h2 = w2o
d_h2_d_zh2 = 1
d_zh2_d_w21 = a1

d_Etotal_d_w21 = d_Etotal_d_w1o * d_w2o_d_h2 * d_h2_d_zh2 * d_zh2_d_w21
#print(f"d_Etotal_d_w21 = {d_Etotal_d_w21}")

d_w2o_d_h2 = w2o
d_h2_d_zh2 = 1
d_zh2_d_w22 = a2

d_Etotal_d_w22 = d_Etotal_d_w1o * d_w2o_d_h2 * d_h2_d_zh2 * d_zh2_d_w22
#print(f"d_Etotal_d_w22 = {d_Etotal_d_w22}")



w1o = w1o - n * d_Etotal_d_w1o
w2o = w1o - n * d_Etotal_d_w2o

w11 = w11 - n * d_Etotal_d_w11
w12 = w12 - n * d_Etotal_d_w12

w21 = w21 - n * d_Etotal_d_w21
w22 = w22 - n * d_Etotal_d_w22
