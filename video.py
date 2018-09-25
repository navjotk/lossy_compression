fig = plt.figure(figsize=(24, 10))
im = plt.imshow(np.transpose(u.data[0,40:-40,40:-40]), animated=True, vmin=-1e0, vmax=1e0, cmap=cm.RdGy, aspect=1.5,
                extent=[origin[0], origin[0] + 1e-3 * dimensions[0] * spacing[0],
                        origin[1] + 1e-3*dimensions[1] * spacing[1], origin[1]])
plt.xlabel('X position (km)',  fontsize=20)
plt.ylabel('Depth (km)',  fontsize=20)
plt.tick_params(labelsize=20)
im2 = plt.imshow(np.transpose(vp), vmin=1.5, vmax=4.5, cmap=cm.jet, aspect=1.5,
                 extent=[origin[0], origin[0] + 1e-3 * dimensions[0] * spacing[0],
                         origin[1] + 1e-3*dimensions[1] * spacing[1], origin[1]], alpha=.4)
import matplotlib.animation as animation

def updatefig(i):
    im.set_array(np.transpose(u.data[i,40:-40,40:-40]))
    im2.set_array(np.transpose(vp))
    return im, im2

ani = animation.FuncAnimation(fig, updatefig, frames=np.linspace(0, nt), blit=True)
ani.save('Adjoint.mp4')
plt.show()
