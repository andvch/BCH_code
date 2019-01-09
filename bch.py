import numpy as np
import random

import matplotlib.pyplot as plt
import time

import gf

class BCH(object):
	
	def __init__(self, n, t):
		q = n.bit_length()
		if (n+1).bit_length() <= q:
			raise ValueError("n != 2^q - 1")
		if t > (n-1)//2:
			raise ValueError("t > (n-1)/2")
		
		poly = open("primpoly.txt").readline()
		poly = poly.replace(" ", "").split(',')
		poly = list(map(int, poly))
		i = np.array(list(map(int.bit_length,poly)))
		i = np.flatnonzero(i-1 == q)
		if i.size == 0:
			raise ValueError("No matching primitive polynomial found in primpoly.txt")
		poly = poly[random.choice(i)]
#		print("Primitive polynomial: " + str(poly) + " ~ " + np.binary_repr(poly))
		self.pm = gf.gen_pow_matrix(poly)
		
		self.R = self.pm[:(2*t), 1]
		self.g = gf.minpoly(self.R, self.pm)[0]
#		print("Generator polynomial:", self.g)
#		print("m = " + str(self.g.size - 1) + "\tk = " + str(n - self.g.size + 1))
	
	def encode(self, U):
		k = U.shape[1]
		V = np.zeros((len(U), len(self.pm)), int)
		V[:,:k] = U
		for i in range(len(V)):
			V[i] = gf.polyadd(V[i], gf.polydiv(V[i], self.g, self.pm)[1])
		return V
	
	def decode(self, W, method='euclid'):
		V = W.astype(object)
		for i in range(len(V)):
			
			s = gf.polyval(V[i], self.R, self.pm)
			for j in s:
				if j != 0:
					break
			else:
				continue
			
			if method == 'pgz':
				e = s.size//2
				while e > 0:
					S = np.empty((e,e),int)
					for j in range(e):
						S[j] = s[j:j+e]
					x = gf.linsolve(S,s[e:2*e],self.pm)
					if not np.any(np.isnan(x)):
						break
					e -= 1
				else:
					V[i,:] = np.nan
					continue
				x = np.append(x, [1])
			
			if method == 'euclid':
				S = np.append(s[::-1], [1])
				x = np.zeros(s.size + 2,int)
				x[0] = 1
				x = gf.euclid(x, S, self.pm, max_deg=s.size//2)[2]
				e = x.size - 1
			
			x = gf.polyval(x, self.pm[:,1], self.pm)
			x = self.pm[np.flatnonzero(x==0), 1]
			
			if method == 'euclid' and x.size != e:
				V[i,:] = np.nan
				continue
			
			x = self.pm[x-1,0] - 1
			V[i,x] = V[i,x]^1
			
			if method == 'pgz':
				s = gf.polyval(V[i], self.R, self.pm)
				for j in s:
					if j != 0:
						V[i,:] = np.nan
						break
			
		return V
	
	def dist(self, check=False):
		k = len(self.pm) - self.g.size + 1
		d = len(self.pm)
		u = np.arange(k - 1, -1, -1).reshape(1,-1)
		for i in range(2**k):
			v = self.encode((i >> u) & 1).reshape(-1)
			if check:
				r = gf.polydiv(v, self.g, self.pm)[1]
				if r.size != 1 or r[0] != 0:
					return False
				s = gf.polyval(v, self.R, self.pm)
				for j in s:
					if j != 0:
						return False
			b = np.sum(v)
			if b < d and b != 0:
				d = b
		if check:
			return d >= self.R.size + 1
		else:
			return d
	
	def checker(self):
		for i in self.g:
			if i != 0 and i != 1:
				return False
		a = np.zeros(len(self.pm)+1, int)
		a[0] = a[-1] = 1
		a = gf.polydiv(a, self.g, self.pm)[1]
		if a.size != 1 or a[0] != 0:
			return False
		return self.dist(check=True)
	



def plot(n,imgname):
	x = np.arange((n-1)//2+1)
	y = np.empty(x.size, float)
	for i in x:
		code = BCH(n, i)
		k = n - code.g.size + 1
		y[i] = k/n
	
	plt.plot(x, y, lw=3)
	plt.xlim (0, (n-1)//2)
	plt.ylim (0, 1)
	plt.xlabel("t")
	plt.ylabel("k/n")
	plt.title("n = " + str(n))
	plt.grid()
	plt.savefig(imgname)
	plt.clf()

def test(code,r,s,inaccuracy=False):
	U = np.random.randint(2, size=(s, len(code.pm) - code.g.size + 1))
	V = code.encode(U)
	if inaccuracy:
		W = np.where(np.random.randint(len(code.pm), size=V.shape) < r, V^1, V)
	else:
		W = V.copy()
		for w in W:
			w[random.sample(range(w.size), r)] ^= 1
	t0 = time.time()
	V1 = code.decode(W, method='pgz')
	t1 = time.time()
	V2 = code.decode(W, method='euclid')
	t2 = time.time()
	
	e1 = np.sum(np.any(V!=V1, axis=1))/s
	e2 = np.sum(np.any(V!=V2, axis=1))/s
	d1 = np.sum([np.isnan(v) for v in V1[:,0]])/s
	d2 = np.sum([np.isnan(v) for v in V2[:,0]])/s
	return t1 - t0, 1 - e1, e1 - d1, d1, t2 - t1, 1 - e2, e2 - d2, d2

if __name__ == "__main__":
	
	print("n\tt\tr\tPGZ time\tOK\tMiss\tError\tEuclid time\tOK\tMiss\tError\n")
	for q in range(2,6):
		n = 2**q - 1
#		plot(n,"p"+str(n)+".png")
		t = n//6
		code = BCH(n, t)
		for r in [0, 1, t, t+1, 3*n//4]:
			print("{0}\t{1}\t{2}\t{d[0]:.6f}\t{d[1]:.2%}\t{d[2]:.2%}\t{d[3]:.2%}\t{d[4]:.6f}\t{d[5]:.2%}\t{d[6]:.2%}\t{d[7]:.2%}".format(n, t, r, d=test(code, r, 100)))
		
	
