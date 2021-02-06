# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 13:13:41 2020

@author: sangbe SIDIBE
"""
import socket
import threading
import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import param_server

#from skimage import io
import matplotlib.cm as cm

# =========================================================================
def read_images (path , sz= (300,300)):
    c = ""
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        
        dirnames.sort()
        for subdirname in dirnames :
            subject_path = os.path.join(dirname, subdirname )
            c = subdirname
            for filename in os.listdir(subject_path ):
                try :
                    im = Image.open(os.path.join (subject_path , filename ))
                    im = im.convert ("L")
                    if (sz is not None ):
                        im = im.resize(sz , Image.ANTIALIAS )
                    X.append (np.asarray (im , dtype =np.uint8 ))
                    y.append (c)
                except IOError :
                    print("I/O error ({0}) : {1} ".format(os.errno , os.strerror ))
                except :
                    print(" Unexpected error :", sys.exc_info() [0])
                    raise
        
    return [X,y]

# =========================================================================
def asRowMatrix (X):
    if len (X) == 0:
        return np.array([])
    mat = np.empty((0 , X [0].size), dtype=X [0].dtype )
    for row in X:
        mat = np.vstack((mat,np.asarray(row).reshape(1,-1)))
    return mat

# =========================================================================
def asColumnMatrix (X):
    if len (X) == 0:
        return np.array ([])
    mat = np.empty ((X [0].size , 0) , dtype =X [0].dtype )
    for col in X:
        mat = np.hstack (( mat , np.asarray ( col ).reshape( -1 ,1)))
    return mat  
    
# =========================================================================
def pca(X, y, num_components =0):
    [n,d] = X.shape
    #[n,d] = np.shape(X)
    if ( num_components <= 0) or ( num_components > n):
        num_components = n
    mu = X.mean ( axis =0)
    X = X - mu
    if n>d:
        C = np.dot (X.T,X)
        [ eigenvalues , eigenvectors ] = np.linalg.eigh (C)
    else :
        C = np.dot (X,X.T)
        [ eigenvalues , eigenvectors ] = np.linalg.eigh (C)
        eigenvectors = np.dot (X.T, eigenvectors )
        for i in range (n):
            eigenvectors [:,i] = eigenvectors [:,i]/ np.linalg.norm ( eigenvectors [:,i])
    # or simply perform an economy size decomposition
    # eigenvectors , eigenvalues , variance = np.linalg.svd (X.T, full_matrices = False )
    # sort eigenvectors descending by their eigenvalue
    idx = np.argsort (- eigenvalues )
    eigenvalues = eigenvalues [idx ]
    eigenvectors = eigenvectors [:, idx ]
    # select only num_components
    eigenvalues = eigenvalues [0: num_components ].copy ()
    eigenvectors = eigenvectors [: ,0: num_components ].copy ()
    return [ eigenvalues , eigenvectors , mu]
    
# =========================================================================
def project (W, X, mu= None ):
    if mu is None :
        return np.dot (X,W)
    return np.dot (X - mu , W)
    
# =========================================================================
def reconstruct (W, Y, mu= None ):
    if mu is None :
        return np.dot(Y,W.T)
    return np.dot (Y,W.T) + mu
        
# =========================================================================
def normalize (X, low , high , dtype = None ):
    X = np.asarray (X)
    minX , maxX = np.min (X), np.max (X)
    # normalize to [0...1].
    X = X - float ( minX )
    X = X / float (( maxX - minX ))
    # scale to [ low...high ].
    X = X * (high - low )
    X = X + low
    if dtype is None :
        return np.asarray (X)
    return np.asarray (X, dtype = dtype )
    
# =========================================================================
def create_font ( fontname ='Tahoma', fontsize =10) :
    return { 'fontname': fontname , 'fontsize': fontsize }
    
# =========================================================================
def subplot (title , images , rows , cols , sptitle =" subplot ", sptitles =[] , colormap =cm.gray , ticks_visible =True , filename = None ):
    fig = plt.figure()
    # main title
    fig.text (.5 ,.95 , title , horizontalalignment ='center')
    for i in range (len( images )):
        ax0 = fig.add_subplot (rows ,cols ,(i +1) )
        plt.setp ( ax0.get_xticklabels () , visible = False )
        plt.setp ( ax0.get_yticklabels () , visible = False )
        if len ( sptitles ) == len ( images ):
            plt.title ("%s #%s" % ( sptitle , str ( sptitles[i])), create_font ('Tahoma',10))
        else :
            plt.title ("%s #%d" % ( sptitle , (i +1)), create_font ('Tahoma',10))
        plt.imshow (np.asarray ( images [i]) , cmap = colormap )
    if filename is None :
        plt.show()
    else :
        fig.savefig( filename )
        
# =========================================================================
class AbstractDistance ( object ):
    
    def __init__(self , name ):
            self._name = name
            
    def __call__(self ,p,q):
        raise NotImplementedError (" Every AbstractDistance must implement the __call__method.")
    @property
    def name ( self ):
        return self._name
    
    def __repr__( self ):
        return self._name

# =========================================================================
class EuclideanDistance ( AbstractDistance ): 
    def __init__( self ):
        AbstractDistance.__init__(self ," EuclideanDistance ")
        
    def __call__(self , p, q):
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        return np.sqrt(np.sum (np.power((p-q) ,2)))

# =========================================================================
class CosineDistance ( AbstractDistance ):
    def __init__( self ):
        AbstractDistance.__init__(self ," CosineDistance ")
        
    def __call__(self , p, q):
        p = np.asarray (p).flatten ()
        q = np.asarray (q).flatten ()
        return -np.dot(p.T,q) / (np.sqrt (np.dot(p,p.T)*np.dot(q,q.T)))
  
# =========================================================================
class BaseModel ( object ):
    def __init__ (self , X=None , y=None , dist_metric = EuclideanDistance () , num_components=0) :
        self.dist_metric = dist_metric
        self.num_components = 0
        self.projections = []
        self.W = []
        self.mu = []
        if (X is not None ) and (y is not None ):
            self.compute (X,y)
            
    def compute (self , X, y):
        raise NotImplementedError (" Every BaseModel must implement the compute method.")
        
    def predict (self , X):
        minDist = np.finfo('float').max
        minClass = -1
        Q = project ( self.W, X.reshape (1 , -1) , self.mu)
        for i in range (len( self.projections )):
            dist = self.dist_metric ( self.projections [i], Q)
            if dist < minDist :
                minDist = dist
                minClass = self.y[i]
        return minClass

# =========================================================================
class EigenfacesModel ( BaseModel ):
    def __init__ (self , X=None , y=None , dist_metric = EuclideanDistance () , num_components=0) :
        super ( EigenfacesModel , self ).__init__ (X=X,y=y, dist_metric = dist_metric , num_components = num_components )
        
    def compute (self , X, y):
        [D, self.W, self.mu] = pca ( asRowMatrix (X),y, self.num_components )
        # store labels
        self.y = y
        # store projections
        for xi in X:
            self.projections.append ( project ( self.W, xi.reshape (1 , -1) , self.mu))            


# =========================================================================
"""
    Client Thread which will be execute on server
"""
class ClientThread(threading.Thread):

    def __init__(self, ip, port, clientsocket):		
        threading.Thread.__init__(self)		
        self.ip = ip
        self.port = port
        self.clientsocket = clientsocket
        print("[+] New thread for %s %s" % (self.ip, self.port, ))
        
    # ----------------------------------------->
    """
        Creation of the thread
    
    """
    """
    def run(self):
        print("Connection of %s %s" % (self.ip, self.port, ))
        cmd = self.clientsocket.recv(1024)    # get the picture's name from client
        print("[SERV recieved]: " , cmd)
        print("[SERV]: opening picture...")
        imtest = Image.open(cmd.decode()) # open the picture
        imtest = imtest.convert ("L")         # convert the picture  
        imtest = imtest.resize((300,300) , Image.ANTIALIAS ) # resize the picture
        test = np.asarray (imtest , dtype =np.uint8 )   # convert as numpy array of uint8
        model = EigenfacesModel (X , y)  # compute Eigen values and vectors 
        print("[SERV]: computing prediction...")
        result = model.predict(test)     # compute the prediction
        print("[SERV result]: ", result)
        self.clientsocket.send(result.encode())
        print("[SERV]: result sent to client")
        print("[SERV]: Client disconnected...")
"""     
    def reception_image(self, path, port, size):
    # fonction qui permet de reconstruire un fichier .jpg a partir de reception de plusieur morceaux
    # le fichier est envoyer en morceaux (ici on prendra 512 octets) pour utiliser le socket
    # qui est limite en taille de donnee a envoyer avec la methode send
        buff = []
        fname = path + str(port) + ".pgm"  #creation d'un chemin
        print("[SERV, filename]: ", fname)
        print ("[SERV]: recieving file...")						#[debug] permet de verifier que l'on ets rentre dans la fontion
        fp = open(fname,'wb')					#creation du nouveau fichier
        try:									#
            for i in range(0,size):				# boucle de 0 à nbr de trame a recevoir
                strng = clientsocket.recv(514)	#lecture trame de 512 octets
                print("[SERV, recieved]: ", type(strng))
                msg = "[OK]: package n°" + str(i) + " recieived!"
                strng = strng.decode()                             
                if strng != "stop\r\n":             # securite: si la trame ets different de 'stop' alors on peut l'ecrire     
                    # strng = strng[0:-2] 
                    buff.append(strng)
                    print ("[SERV, lengthPackage:n°packege]: ", str(len(strng))+":"+str(i))                        
                    fp.write(strng.encode())				# ecriture de la trame
                    clientsocket.send('[OK]: little package\r\n'.encode())		# acuse de reception
                    print(msg)
                # i += 1							#increment boucle
            fp.close()							# fermeture du fichier quand c'est fini
            clientsocket.send('[OK]: fin\r\n'.encode())			#envoie message de fin
            # img = Image.open(fname)
            # plt.plot(img) #--------- Add
            print('[SERV, Buffer Type]: ', type(buff))
            # arr = np.asarray(buff, dtype=np.uint8)
            # img = Image.fromarray(arr)
            # img.save("E:\IAI5\PROJ942\Image_Test\test.jpg")            
            
            print ("[SERV, OK]: Data Received successfully")		#[debug] permet de verifier l'ecriture
        except:
            os.remove(path + str(port) + ".jpg")#securite : si il y a un probleme, supression du fichier 
            print ("[SERV, WARNING]: il y a eu un probleme")		#[debug] permet d'informer en cas de probleme
        
    def run(self):
        #fonction realise lors de la creation du thread
        print("Connection de %s %s" % (self.ip, self.port, ))
        r = self.clientsocket.recv(1024)    #reception commande
        r = r.decode() # Decoder la commande
        print ("[SERV, cmd recieved]: ", r.encode())      
        print ("[SERV, cmd length]: ", len(r))    					# info sur la taille de reception				
        if r[0] == '1':						# test de reception
#            try:							# try/exept permete la supression du fichier cree en cas probleme
            self.clientsocket.send("[OK]: cmd recieved\r\n".encode())# confirmation reception commande
            size = self.clientsocket.recv(1024)# attente du nombre de paquet a recevoir pour l'image ( de 512 octets)
            size = size.decode()
            print ("[SERV, package length]: ", type(size))
            # size = size[0:-2]
            	
            self.clientsocket.send('[OK]: package length\r\n'.encode())# confirmation reception taille
            isize = int(size)			# cast en int de la taille pour utilisation dans une boucle
            self.reception_image(param_server.img_test_path, self.port, isize) # fonction de reception           
            print("[SERV]: picture recieved")	#[debug] permet de verifier la reception
            imtest = imtest.convert ("L")         # convert the picture  
            imtest = imtest.resize((300,300) , Image.ANTIALIAS ) # resize the picture
            test = np.asarray (imtest , dtype =np.uint8 )   # convert as numpy array of uint8
            model = EigenfacesModel (X , y)  # compute Eigen values and vectors 
            print("[SERV]: computing prediction...")
            result = model.predict(test)     # compute the prediction
            print("[SERV result]: ", result)
            self.clientsocket.send(result.encode())
            print("[SERV]: result sent to client")
            print("[SERV]: Client disconnected...")
            
            # resize( self.port)			# formate l'image recue
            # print("resized")			#[debug] permet de verifier que l'image a bien ete redimensionnee
            # res = calculRessemblance(self.port)# calcul l aressemblance
            # print("calculated")   		#[debug] permet de verifier l'execution de la fonction
            # self.clientsocket.send(str(res))# envoie la reponse au client
            # print (res)					#[debug] permet de verifier le reultat
        else:
            cmd = r
            print("[SERV]: opening picture...")
            imtest = Image.open(cmd) # open the picture
            imtest = imtest.convert ("L")         # convert the picture  
            imtest = imtest.resize((300,300) , Image.ANTIALIAS ) # resize the picture
            test = np.asarray (imtest , dtype =np.uint8 )   # convert as numpy array of uint8
            model = EigenfacesModel (X , y)  # compute Eigen values and vectors 
            print("[SERV]: computing prediction...")
            result = model.predict(test)     # compute the prediction
            print("[SERV result]: ", result)
            self.clientsocket.send(result.encode())
            print("[SERV]: result sent to client")
            print("[SERV]: Client disconnected...")
#            except:
#                os.remove("image/image"+str(port)+".jpg")# suppression du fichier cree en cas de probleme
#                print "il y a eu un probleme"#[debug] permet d'informer en cas de probleme
        print("[SERV]: Client déconnecté...")
        exit()
# -- end class

# =========================================================================
#           INITIALIZING THE SERVER
# =========================================================================
tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcpsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

tcpsock.bind((param_server.ip_server, param_server.port_server))

# Load all images from database
[X,y] = read_images (param_server.db_img_path)
print(asRowMatrix(X).shape)

while True:
    tcpsock.listen(10)
    print("Listenning...")
    (clientsocket, (ip, port)) = tcpsock.accept()
    newthread = ClientThread(ip, port, clientsocket)
    newthread.start()

