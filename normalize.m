function x =  normalize(x)

z = sum(x(:));
z(z==0) = 1;
x = x./z;

end