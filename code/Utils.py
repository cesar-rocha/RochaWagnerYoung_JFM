#
# Utility functions used in simulation and plotting
#   scripts
#

import matplotlib.pyplot as plt

def save_parameters(model):
    """ Save simulation parameters """
    fno = model.path+"/parameters.h5"
    h5file = h5py.File(fno, 'w')

    h5file.create_dataset("dimensional/Ue", data=(Ue))
    h5file.create_dataset("dimensional/Uw", data=(Uw))
    h5file.create_dataset("dimensional/ke", data=(ke))
    h5file.create_dataset("dimensional/m", data=(m))
    h5file.create_dataset("dimensional/N", data=(N))
    h5file.create_dataset("dimensional/f0", data=(f0))
    h5file.create_dataset("dimensional/Te", data=(Te))
    h5file.create_dataset("dimensional/L", data=(L))
    h5file.create_dataset("dimensional/nu4", data=(nu4))
    h5file.create_dataset("dimensional/nu4w", data=(nu4w))
    h5file.create_dataset("nondimensional/nx", data=(nx))
    h5file.create_dataset("nondimensional/alpha", data=(alpha))
    h5file.create_dataset("nondimensional/hslash", data=(hslash))
    h5file.create_dataset("nondimensional/Ro", data=(Ro))

    h5file.close()


def plot_fig_label(ax, xc=.95, yc=0.075 ,label="a"):
    """ Plot label numbering for multi-panel figures """
    plt.text(0.95, 0.1,label,
                horizontalalignment='center',
                verticalalignment='center',
                transform = ax.transAxes,bbox=dict(boxstyle='circle',facecolor='white'))
