# -*- coding: utf-8 -*-

from CoffeeLibs.criteres import diff_grad_R,Dregule
from CoffeeLibs.pzernike import pmap, zernike, pzernike
from CoffeeLibs.tools    import circle
import matplotlib.pyplot as plt
import numpy as np

# %% Compare grad R

w = 64

[Ro,Theta]  =  pmap(w,w)
coeff       = 70/np.arange(1,20) # Coeff to generate point
coeff[0:6]  = [0,0,0,0,0,0]

pup    = circle(w,w,(w)//2)
point  = ( pzernike(Ro,Theta,coeff) + np.random.normal(0, 0, (w,w)) )

grad_diff_up_sob  = diff_grad_R(point,pup,"sobel") / 100
grad_diff_up_np   = diff_grad_R(point,pup,"np")
grad_analytic_up  = Dregule(point,pup=pup)



plt.figure(1)
plt.gcf().subplots_adjust(hspace = 0.5)

plt.suptitle("Comparaison gradient (couper pour mieux voir la différence hors effet de bords)")

plt.subplot(3,5,1),plt.imshow(grad_analytic_up,cmap='jet'),plt.title("Garident Analytique Regule"),plt.colorbar()
plt.subplot(3,5,2),plt.imshow(grad_diff_up_np,cmap='jet'),plt.title("Garident difference NP"),plt.colorbar()
plt.subplot(3,5,3),plt.imshow((grad_diff_up_sob),cmap='jet'),plt.title("Garident difference SOBEL"),plt.colorbar()
plt.subplot(3,5,4),plt.imshow((grad_diff_up_np-grad_analytic_up),cmap='jet'),plt.title("Erreur NP"),plt.colorbar()
plt.subplot(3,5,5),plt.imshow((grad_diff_up_sob-grad_analytic_up),cmap='jet'),plt.title("Erreur sobel"),plt.colorbar()

e = grad_diff_up_np/grad_analytic_up
print( np.mean( e[np.isfinite(e)] ) )

grad_diff_up_np   *=  circle(w,w,(w//2)-3)
grad_diff_up_sob  *=  circle(w,w,(w//2)-3)
grad_analytic_up  *=  circle(w,w,(w//2)-3)

plt.subplot(3,5,6),plt.imshow(grad_analytic_up,cmap='jet'),plt.title("Garident Analytique Regule coupé"),plt.colorbar()
plt.subplot(3,5,7),plt.imshow(grad_diff_up_np,cmap='jet'),plt.title("Garident difference NP"),plt.colorbar()
plt.subplot(3,5,8),plt.imshow((grad_diff_up_sob),cmap='jet'),plt.title("Garident difference SOBEL"),plt.colorbar()
plt.subplot(3,5,9),plt.imshow((grad_diff_up_np-grad_analytic_up),cmap='jet'),plt.title("Erreur NP"),plt.colorbar()
plt.subplot(3,5,10),plt.imshow((grad_diff_up_sob-grad_analytic_up),cmap='jet'),plt.title("Erreur SOBEL"),plt.colorbar()

plt.subplot(3,5,13),plt.imshow(point,cmap='jet'),plt.title("point de plus haute amplitude"),plt.colorbar()

plt.show()
