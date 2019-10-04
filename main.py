import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd


def determine_shape(body):
    evals, evecs = find_principal_axes(body.astype(np.float64))
    angle = np.arctan(evecs[1,np.argmax(evals)]/evecs[0,np.argmax(evals)])
    aligned_body = nd.rotate(body,np.rad2deg(angle))
    #aligned_body[aligned_body < 1e-3] = 0
    aligned_body = aligned_body.astype(np.int)


    length, height = determine_dimensions(aligned_body)

    return (length,height,-np.rad2deg(angle))


def determine_dimensions(body):
    """"
    https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/
    """

    body[body<1e-2] = 0
    y, x = np.nonzero(body)
    y = np.float64(y)
    x = np.float64(x)


    length = x.max()-x.min()
    height = y.max()-y.min()

    if length<200:
        plt.imshow(body)
        plt.show()

    return length, height


def insert_object(body, domain, num=1, min_spacing=10):
    n, m = domain.shape
    n_inserted = 0

    shapes_inserted = []

    for i in range(num):
        thing_dil, shape = body()
        thing = thing_dil.copy()

        if min_spacing > 0:
            thing_dil = np.pad(thing_dil, min_spacing, mode="constant")
            thing_dil = nd.binary_dilation(thing_dil, iterations=min_spacing)




        thing_dil_n, thing_dil_m = thing_dil.shape
        thing_n, thing_m = thing.shape

        pos_n = int(np.random.rand() * (n - thing_dil_n))
        pos_m = int(np.random.rand() * (m - thing_dil_m))

        covered_domain = domain[pos_n:pos_n + thing_dil_n, pos_m:pos_m + thing_dil_m]
        collision = np.logical_and(thing_dil, covered_domain)

        overlap = np.any(collision)

        if overlap == False:
            n_inserted += 1
            domain[pos_n + min_spacing:pos_n + min_spacing + thing_n,
            pos_m + min_spacing:pos_m + min_spacing + thing_m] = domain[
                                                                 pos_n + min_spacing:pos_n + min_spacing + thing_n,
                                                                 pos_m + min_spacing:pos_m + min_spacing + thing_m] + thing
            shapes_inserted.append(shape)

    print("Inserted %i object into the microstructure" % n_inserted)
    return domain, np.array(shapes_inserted)


def find_principal_axes(body):
    """"
    https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/
    """

    body[body < 1e-3] = 0

    y, x = np.nonzero(body)
    y = np.float64(y)
    x = np.float64(x)

    x = x - np.mean(x)
    y = y - np.mean(y)
    coords = np.vstack([x, y])

#     plt.plot(x, y, 'o')
#     plt.show()

    cov = np.cov(coords)

    evals, evecs = np.linalg.eig(cov)

    return evals, evecs


class Block(object):
    def __init__(self, length, aspect, angle):
        self.length = length
        self.aspect = aspect
        self.angle = angle

    def __call__(self):
        length = np.random.randint(self.length[0], self.length[1])
        aspect = np.random.randint(self.aspect[0], self.aspect[1])
        angle = np.random.rand() * (self.angle[1] - self.angle[0]) + self.angle[0]

        rect = np.ones((length, length * aspect))
        return nd.rotate(rect, angle), (length*aspect,aspect,angle)


class MicroStructure(object):
    # This object holds characteristics of a micro structure
    def __init__(self, lengt_dist, aspect_dist, angle_dist, n_objs, min_spacing):
        self.length_dist = lengt_dist
        self.aspect_dist = aspect_dist
        self.angle_dist = angle_dist
        self.n_objects = n_objs
        self.min_spacing = min_spacing


def characterise_microstructure(domain):
    domain_bin = domain.astype(np.bool)
    shape = []

    labeled_domain,n = nd.label(domain_bin)
    features = nd.find_objects(labeled_domain)


    for i,feature in enumerate(features):
        feature_domain = labeled_domain[feature].astype(np.int)
        # Remove any other features within the frame
        feature_domain[feature_domain!=i+1] = 0

        shape.append(determine_shape(feature_domain.astype(np.bool)))

    shape = np.array(shape)



    length_dist = np.histogram(shape[:,0])
    aspect_dist = np.histogram(shape[:,0]/shape[:,1])



    print(length_dist)

    return MicroStructure(lengt_dist=shape[:,0], aspect_dist=shape[:,0]/shape[:,1], angle_dist=shape[:,2], n_objs=n, min_spacing=None)


def run():
    domain_size = (5000, 5000)
    domain = np.zeros(domain_size)

    # geom = Block(length=(10, 20), aspect=(5, 10), angle=(0., 0))
    geom = Block(length=(60, 80), aspect=(5, 6), angle=(45, 45))
    n_geoms = 8000

    domain_with_stuff,stats_true = insert_object(geom, domain, num=n_geoms, min_spacing=20)




    stats_guess = characterise_microstructure(domain_with_stuff)

    plt.hist(stats_true[:,0],bins=50,label="correct",alpha=0.7)
    plt.hist(stats_guess.length_dist,bins=50,label="guessed",alpha=0.7)
    plt.legend()
    plt.show()

    plt.imshow(domain_with_stuff, cmap=plt.cm.gray)
    plt.show()
    # plt.imsave("struktur.png",domain_with_stuff,cmap=plt.cm.gray)


if __name__ == '__main__':
    run()
