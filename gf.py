import numpy as np

def gen_pow_matrix(primpoly):
	q = primpoly.bit_length() - 1
	pm = np.empty((2**q - 1, 2), int)
	t = 1
	for i in range(2**q - 1):
		t <<= 1
		if (t >> q) != 0:
			t ^= primpoly
		pm[i,1] = t
		pm[t-1,0] = i+1
	return pm

def add(A, B):
	return (A^B)

def sum(X, axis=0):
	return np.bitwise_xor.reduce(X, axis=axis).reshape([(-1),(-1,1)][axis])

def prod(X, Y, pm):
	return np.where(np.logical_and(X != 0, Y != 0), pm[(pm[X-1,0] + pm[Y-1,0] - 1) % len(pm), 1], 0)

def divide(X, Y, pm):
	if (0 in Y):
		raise ZeroDivisionError
	return np.where(X != 0, pm[(pm[X-1,0] - pm[Y-1,0] - 1) % len(pm), 1], 0)

def linsolve(A, b, pm):
	C = np.column_stack((A, b))
	for i in range(len(A)):
		S = C[i:,i:]
		n = np.flatnonzero(S[:,0])
		if n.size == 0:
			return np.nan
		if S[0,0] == 0:
			S[[0,n[0]]] = S[[n[0],0]]
		if n.size == 1:
			continue
		
		t = divide(S[n[1:],0], np.full(n.size-1, S[0,0], int), pm)
		t = np.full((S[0].size,t.size), t, int).transpose()
		S[n[1:]] = add(S[n[1:]], prod(S[0], t, pm))
	
	x = np.empty_like(A[0])
	x[-1] = divide(C[-1,-1].reshape(1), C[-1,-2].reshape(1), pm)
	for i in range(len(A)-2, -1, -1):
		t = sum(prod(C[i,i+1:-1], x[i+1:], pm))
		x[i] = divide(add(np.array([t]), C[i,-1].reshape(1)), C[i,i].reshape(1), pm)
	return x

def minpoly(x, pm):
	p = np.array([1])
	roots = np.zeros(len(pm)+1, bool)
	for a in x:
		while not roots[a]:
			p = polyprod(p, np.array([1,a]), pm)
			roots[a] = True
			if a == 0:
				break
			a = pm[2*pm[a-1,0] % len(pm) - 1, 1]
	return p, np.flatnonzero(roots)

def clean_zeros(p):
	if p[0] != 0:
		return p
	n = np.flatnonzero(p)
	i = p.size-1 if n.size == 0 else n[0]
	return p[i:]

def polyval(p, x, pm):
	p = clean_zeros(p)
	y = np.full_like(x, p[-1])
	v = np.ones_like(x)
	for k in p[-2::-1]:
		v = prod(v, x, pm)
		y = add(y, prod(np.full_like(v, k), v, pm))
	return y

def polyadd(p1, p2):
	if p1.size == p2.size:
		return add(p1, p2)
	elif p1.size > p2.size:
		return add(p1, np.append(np.zeros(p1.size-p2.size, int), p2))
	else:
		return add(np.append(np.zeros(p2.size-p1.size, int), p1), p2)

def polyprod(p1, p2, pm):
	p1 = clean_zeros(p1)
	p2 = clean_zeros(p2)
	if p1.size < p2.size:
		p1, p2 = p2, p1
	y = np.array([],int)
	for k in p2:
		y = polyadd(np.append(y, [0]), prod(np.full_like(p1, k), p1, pm))
	return clean_zeros(y)

def polydiv(p1, p2, pm):
	p1 = clean_zeros(p1)
	p2 = clean_zeros(p2)
	if p1.size < p2.size:
		return np.array([0]), p1.copy()
	a = np.empty(p1.size - p2.size + 1, int)
	b = p1.copy()
	for i in range(a.size):
		a[i] = divide(b[i].reshape(1), p2[0].reshape(1), pm)
		b[i:i+p2.size] = add(b[i:i+p2.size], prod(np.full_like(p2, a[i]), p2, pm))
	return a, clean_zeros(b)

def euclid(p1, p2, pm, max_deg=0):
	p1 = clean_zeros(p1)
	p2 = clean_zeros(p2)
	x0, x = np.array([1]), np.array([0])
	y0, y = np.array([0]), np.array([1])
	r = p2
	while r.size > max_deg + 1:
		q, r = polydiv(p1, p2, pm)
		x0, x, y0, y = x, x0, y, y0
		x = polyadd(x, polyprod(x0, q, pm))
		y = polyadd(y, polyprod(y0, q, pm))
		p1, p2 = p2, r
	return r, x, y
