# import all functions from the tkinter 

from shapely.geometry import Point, Polygon, LinearRing, LineString
from shapely.geometry import MultiLineString, MultiPoint, GeometryCollection
from shapely.geos import TopologicalError
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from logging import error
from math import atan

import tkinter as tk
import tkinter.font as font
import tkFont
#import mahotas
import tkinter,yaml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import * 
import threading
import serial
import time
##from ScrolledText import *
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import matplotlib.axes as ax
import time
from matplotlib import pyplot as plt
from shapely.geometry.polygon import Polygon
from descartes import PolygonPatch
import matplotlib as mpl
from tkinter import ttk
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from math import sin, cos, radians, pi
from collections import OrderedDict
import ImageUtils
import matplotlib.animation as animation
import tqdm

import random
import os
##import Image
##import Queue
##import StringIO
from threading import Thread, Lock
##from PIL import ImageTk
###import mapTool
##import urlFetcher
##from PIL import Image
import cv2
import numpy as np
import math
from math import sqrt, pow
from math import sin, cos, sqrt, atan2, radians
import socket
import sys
udp_socket = None
socket1 = None 

#sudo python swarm_20_quad_01.py
##from automission import automission

import math
import numpy as np
import pandas as pd
from scipy import interpolate

import geopy
from geopy.distance import geodesic
import argparse

import datetime

radius_of_earth = 6378100.0 # in meters

import os
import sys, csv
from dronekit import connect, VehicleMode, LocationGlobalRelative, Command, LocationGlobal
from pymavlink import mavutil
#from ubidots import ApiClient
import argparse 
import json
import random
import serial
import socket, struct, time
from math import radians,cos,sin,asin,sqrt,pi
#import pygame
#from modules.utils import *
import threading
##import Queue
import subprocess
from matplotlib.figure import Figure	
import tkinter.filedialog

all_follower_status = []

home_loc_all =[]

rtl_moniter_flag = False

takeoff_flag = False
RTH_flag = False

search_flag = False


##global serial_object

data123 = False
lat = None
lon = None
heading = None


width=640
height=640
center =[]

MAP_HEIGHT = 640
MAP_WIDTH = 640
SCALE = 1
ZOOM = 19

length = 100
flag123 = False
circle_pos_flag = False
master_ip="192.168.6.151"
'''
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('192.168.6.210', 10010)

sock_tx1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  #...send to gps data to GCS
server_address_tx1 = ('192.168.6.210', 13697)

sock_rx1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)   # receive command from GCS
server_address_rx1 = ('', 10243)
sock_rx1.bind(server_address_rx1)
'''
print(socket.__file__)
#udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#server_address1 = ('192.168.6.151', 12008)
#udp_socket2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#server_address2 = ('192.168.6.152', 12008)
udp_socket3 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address3 = ('192.168.6.153', 12008)
udp_socket4 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address4 = ('192.168.6.154', 12008)
udp_socket5 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address5 = ('192.168.6.155', 12008)
#udp_socket6 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#server_address6 = ('192.168.6.156', 12008)
#udp_socket7 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#server_address7 = ('192.168.6.157', 12008)
#udp_socket8 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#server_address8 = ('192.168.6.158', 12008)
#udp_socket9 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#server_address9 = ('192.168.6.159', 12008)
udp_socket10 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address10 = ('192.168.6.160', 12008)

#udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#server_address1 = ('192.168.6.151', 12008)
#udp_socket2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#server_address2 = ('192.168.6.152', 12008)
share_data_udp_socket3 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
share_data_server_address3 = ('192.168.6.153', 12008)
share_data_udp_socket4 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
share_data_server_address4 = ('192.168.6.154', 12008)
share_data_udp_socket5 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
share_data_server_address5 = ('192.168.6.155', 12008)
#udp_socket6 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#server_address6 = ('192.168.6.156', 12008)
#udp_socket7 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#server_address7 = ('192.168.6.157', 12008)
#udp_socket8 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#server_address8 = ('192.168.6.158', 12008)
#udp_socket9 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#server_address9 = ('192.168.6.159', 12008)
share_data_udp_socket10 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
share_data_server_address10 = ('192.168.6.160', 12008)

#socket1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#server_address11 = ('192.168.6.151', 12002)
#socket2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#server_address12 = ('192.168.6.152', 12002)
socket3 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address13 = ('192.168.6.153', 12002)
socket4 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address14 = ('192.168.6.154', 12002)
socket5 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address15 = ('192.168.6.155', 12002)
#socket6 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#server_address16 = ('192.168.6.156', 12002)
#socket7 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#server_address17 = ('192.168.6.157', 12002)
#socket8 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#server_address18 = ('192.168.6.158', 12002)
#socket9 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#server_address19 = ('192.168.6.159', 12002)
socket10 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address20 = ('192.168.6.160', 12002)

#file_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#file_server_address = ('192.168.6.151', 12003)  
#file_sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#file_server_address1 = ('192.168.6.152', 12003)
file_sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
file_server_address2 = ('192.168.6.153', 12003)
file_sock3 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
file_server_address3 = ('192.168.6.154', 12003)
file_sock4 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
file_server_address4 = ('192.168.6.155', 12003)
#file_sock5 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#file_server_address5 = ('192.168.6.156', 12003)
#file_sock6 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#file_server_address6 = ('192.168.6.157', 12003)
#file_sock7 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#file_server_address7 = ('192.168.6.158', 12003)
#file_sock8 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#file_server_address8 = ('192.168.6.159', 12003)
file_sock9 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
file_server_address9 = ('192.168.6.160', 12003)


#mavlink_sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#mavlink_server_address1 = ('192.168.6.138', 12045)

#mavlink_sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#mavlink_server_address2 = ('192.168.6.138', 12046)

mavlink_sock3 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
mavlink_server_address3 = ('192.168.6.153', 12045)

mavlink_sock4 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
mavlink_server_address4 = ('192.168.6.154', 12045)

mavlink_sock5 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
mavlink_server_address5 = ('192.168.6.155', 12045)

#mavlink_sock6 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#mavlink_server_address6 = ('192.168.6.138', 12050)

#mavlink_sock7 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#mavlink_server_address7 = ('192.168.6.138', 12051)

#mavlink_sock8 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#mavlink_server_address8 = ('192.168.6.138', 12052)

#mavlink_sock9 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#mavlink_server_address9 = ('192.168.6.138', 12053)

mavlink_sock10 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
mavlink_server_address10 = ('192.168.6.160', 12045)

from MAVProxy.modules.mavproxy_map import mp_elevation

EleModel = mp_elevation.ElevationModel()

##from automission import automission
import math
from math import sqrt, pow
from math import sin, cos, sqrt, atan2, radians
import datetime
import os
import sys, csv
from dronekit import connect, VehicleMode, LocationGlobalRelative, Command, LocationGlobal
from pymavlink import mavutil
#from ubidots import ApiClient
import argparse 
import json
import random
import serial
import socket, struct, time
from math import radians,cos,sin,asin,sqrt,pi
#import pygame
#from modules.utils import *
import threading
##import Queue
import subprocess
from tkinter import OptionMenu
import os, sys
from math import sin, cos, radians, pi
from pyproj import Proj, transform

import time, cv2, os
#import imagezmq
import cv2
import time
import threading
import socket               # Import socket module
import sys,time, threading
#....import fractions, pyexiv2, math

import shutil

path_01 = '/home/muthu/swarm_GCS_new/messi/Receive_image/'

radius_of_earth = 6378100.0 # in meters

fleast_m = Proj(init='epsg:3857')
wgs84 = Proj(proj='latlong',datum='WGS84')
M2FT = 3.2808399
FT2M = (1.0/M2FT)

loc = []
loc_01 = []


original_location = []
import argparse
R = 6373.0

count_123 = 0

aggr_and_rtl_flag = False

home_location = None

home_location = []

RTH_array = [100,100,100,100,100,100,100,100]


heartbeat_timeout_data = []

follower_host_tuple = []
uavs_array=[]
missionlist_uav1 = []
missionlist_uav2 = [] 
missionlist_uav3 = []
missionlist_uav4 = []
missionlist_uav5 = []
missionlist_uav6 = []
missionlist_uav7 = []
missionlist_uav8 = []
missionlist_uav9 = []
missionlist_uav10 = []
missionlist_uav11 = []
missionlist_uav12 = []
missionlist_uav13 = []
missionlist_uav14 = []
missionlist_uav15 = []
missionlist_uav16 = []
missionlist_uav17 = []
missionlist_uav18 = []
missionlist_uav19 = []
missionlist_uav20 = []
missionlist_uav21 = []
missionlist_uav22 = []
missionlist_uav23 = []
missionlist_uav24 = []
missionlist_uav25 = []
missionlist_uav_all = []

follower_host_tuple_G1 = []
follower_host_tuple_G2 = []
follower_host_tuple_G3 = []
follower_host_tuple_G4 = []
follower_host_tuple_G5 = []
counter_G1 = []
counter_G2 = [] 
counter_G3 = []
counter_G4 = []
counter_G5 = []

wp_pos = []

goto_lat_g = None
goto_lon_g = None
goto_alt_g = None

#ip_status = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]

ip_status = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]

self_heal = []

wp_navigation_guided_forward_flag = False
wp_navigation_guided_return_flag = False

wp_navigation_stop_forward_flag = False
wp_navigation_stop_return_flag = False

#self_heal_manual_flag = False

##helv35=font.Font(family='Times New Roman', size=15)


from imutils import build_montages
from datetime import datetime
import numpy as np
#import imagezmq
import argparse
import imutils
import cv2

import tkinter

check_box_flag1 = False
check_box_flag2 = False
check_box_flag3 = False
check_box_flag4 = False
check_box_flag5 = False

control_command = False

frame1 = None
frame2 = None

master = 1
no_uavs = 25

# initialize the ImageHub object
#imageHub = imagezmq.ImageHub()

RTL_all_flag = False
frameDict = {}

# initialize the dictionary which will contain  information regarding
# when a device was last active, then store the last time the check
# was made was now
lastActive = {}
lastActiveCheck = datetime.now()

# stores the estimated number of Pis, active checking period, and
# calculates the duration seconds to wait before making a check to
# see if a device was active
ESTIMATED_NUM_PIS = 4
ACTIVE_CHECK_PERIOD = 10
ACTIVE_CHECK_SECONDS = ESTIMATED_NUM_PIS * ACTIVE_CHECK_PERIOD

mW = 2
mH = 3


import locatePosition as locate
import rrtStarCollision as rsc
import time, math, random
import patterns as pt
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

## = 5
all_pos_data =  [(500,500), (500,500), (500,500), (500,500), (500, 500), (500, 500), (500, 500), (500, 500), (500, 500), (500, 500), (500, 500), (500, 500)]


xy_pos = []
latlon_pos = []

slave_heal_ip = ['192.168.6.101', '192.168.6.102', '192.168.6.103','192.168.6.104','192.168.6.105','192.168.6.106','192.168.6.107','192.168.6.108','192.168.6.109','192.168.6.110','192.168.6.111', '192.168.6.112', '192.168.6.113', '192.168.6.114','192.168.6.115','192.168.6.116','192.168.6.117','192.168.6.118','192.168.6.119','192.168.6.120','192.168.6.121','192.168.6.122','192.168.6.123','192.168.6.124','192.168.6.125']

ch_e2 = False
ch_e3 = False
ch_e4 = False
ch_e5 = False
ch_e6 = False
ch_e7 = False
ch_e8 = False
ch_e9 = False
ch_e10 = False

slave_odd_lost = ['192.168.6.103', '192.168.6.105', '192.168.6.107', '192.168.6.109',]
slave_even_lost = ['192.168.6.102', '192.168.6.104', '192.168.6.106', '192.168.6.108', '192.168.6.110']

wp_navigation_flag = False
wp_navigation_return_flag = False

import Queue
import numpy as np
import geopy
from geopy.distance import vincenty
import socket
import threading
import multiprocessing
import os

# Constant parameters.
factor_deg2rad = math.pi/180.0
self_heal_move_flag = False

search_no_time = 0

fileloc="/home/dhaksha/Documents/strike_socket/dce_swarm_nav/swarm_tasks-main/swarm_tasks/Examples/basic_tasks/"
print("fileloc",fileloc)
filename=""
filepath=""
	    
class Rtf:
    """Rotation transform"""
    def __init__(self, angle):
        self.angle = angle
        self.w = np.radians(90 - self.angle)
        self.irm = np.mat([[np.cos(self.w), -np.sin(self.w), 0.0],
                           [np.sin(self.w), np.cos(self.w),  0.0],
                           [0.0,         0.0,          1.0]])
    
class AreaPolygon:
    """Polygon object definition for area coverage path planning"""

    def __init__(self, coordinates, initial_pos, angle, interior=[], ft=5.0):
        """Initialization of polygons and transforms"""
        self.P = Polygon(coordinates, interior)

        # Compute path angle
        if 0 <= angle <=360:
            self.rtf = Rtf(angle) # base on provided angle
        elif angle > 360:
            self.rtf = self.rtf_longest_edge() # based on longest edge of polygon
        else:
            self.rtf = self.rtf_longest_edge() # based on longest edge of polygon
        self.rP = self.rotated_polygon()    
            
        # Determine origin (i.e. closest vertex to current position)
        self.origin = self.get_furthest_point(self.P.exterior.coords, initial_pos)[0]
        print('Origin: ({}, {})'.format(self.origin[0], self.origin[1]))
        self.ft = ft


    def rtf_longest_edge(self):
        """Computes rotation transform based on longest edge"""

        # Find the longest edge coordinates and angle
        coords = list(self.P.exterior.coords)
        num = len(coords)
        distances = [Point(coords[(num-i)%num]).distance(Point(coords[(num-i-1)%num])) for i in range(num)][::-1]
        max_index = distances.index(max(distances))
        
        dy = float(coords[max_index][1] - coords[max_index + 1][1])
        dx = float(coords[max_index][0] - coords[max_index + 1][0])
        
        return Rtf(np.degrees(atan(dy/dx)))

    def rotate_points(self, points):
        """Applies rtf to polygon coordinates"""
        new_points = []
        for point in points:
            point_mat = np.mat([[point[0]],[point[1]],[0]], dtype='float64')
            new_point = self.rtf.irm * point_mat
            new_points.append(np.array(new_point[:-1].T, dtype='float64'))
        return np.squeeze(np.array(new_points, dtype='float64'))

    def rotate_from(self, points):
        """Rotate an ndarray of given points(x,y) from a given rotation"""
        if type(points) != np.ndarray:
            raise TypeError("rotate_from: takes an numpy.ndarray")
        new_points = []
        for point in points:
            point_mat = np.mat([[point[0]],[point[1]],[0]], dtype='float64')
            new_point = self.rtf.irm.I * point_mat
            new_points.append(np.array(new_point[:-1].T, dtype='float64'))
        return np.squeeze(np.array(new_points, dtype='float64'))

    def rotated_polygon(self):
        """Applies rtf to polygon and holes (if any)"""
        points = np.array(self.P.exterior)
        tf_points = self.rotate_points(points)
        tf_holes = []
        for hole in self.P.interiors:
            tf_holes.append(self.rotate_points(np.array(hole)))
        return self.array2polygon(tf_points, tf_holes)

    def array2polygon(self, points, holes):
        new_exterior = []
        new_interior = []
        
        # Exterior
        for point in points:
            new_exterior.append((float(point[0]),float(point[1])))
            
        # Interior
        for hole in holes:
            new_hole = []
            for p in hole:
                new_hole.append((float(p[0]), float(p[1])))
            new_interior.append(new_hole)
        return Polygon(new_exterior, new_interior)

    def generate_path(self):
        """Generate parallel coverage path lines"""
        starting_breakdown = self.rP.bounds[0:2]  # poly.bounds = bounding box
        line = LineString([starting_breakdown, (starting_breakdown[0],
                                                starting_breakdown[1] +
                                                self.rP.bounds[3] - self.rP.bounds[1])])
        try:
            bounded_line = self.rP.intersection(line)
        except TopologicalError as e:
            error("Problem looking for intersection.", exc_info=1)
            return
        
        lines = [bounded_line]
#         iterations = int(ceil((self.rP.bounds[2] - self.rP.bounds[0]) / ft)) + 1
        iterations = int((self.rP.bounds[2] - self.rP.bounds[0]) / self.ft) + 2
        for x in range(1, iterations):
            bounded_line = line.parallel_offset(x * self.ft, 'right')
            if self.rP.intersects(bounded_line):
                try:
                    bounded_line = self.rP.intersection(bounded_line)
                except TopologicalError as e:
                    error("Problem looking for intersection.", exc_info=1)
                    continue
                lines.append(bounded_line)
        return lines

    def sort_points(self, point, liste):
        "Sorts a set of points by distance to a point"
        liste.sort(lambda x, y: cmp(x.distance(Point(*point)),
                                y.distance(Point(*point))))
        return liste

    def get_furthest_point(self, ps, origin):
        "Get a point along a line furthest away from a given point"
        orig_point = Point(*origin)
        return sorted(ps, lambda x, y: cmp(orig_point.distance(Point(*x)),orig_point.distance(Point(*y))))


    def order_points(self, lines, initial_origin):
        "Return a list of points in a given coverage path order"
        origin = initial_origin
        results = []
        while True:
            if not len(lines):
                break
            self.sort_points(origin, lines)
            f = lines.pop(0)
            if type(f) == GeometryCollection:
                continue
            if type(f) == MultiLineString:
                for ln in f:
                    lines.append(ln)
                continue
            if type(f) == Point or type(f) == MultiPoint:
                continue
            xs, ys = f.xy
            ps = zip(xs, ys)
            (start, end) = self.get_furthest_point(ps, origin) # determine direction of path (which of the 2 coordinates is closest to the previous endpoint)
            results.append(origin)
            # results.append(start)
            results.append(start)
            # results.append(end)
            origin = end
        return results

    # NOTE: the decomposition of the area will depend on the robot's footprint
    def boustrophedon_decomposition(self, origin):
        """Decompose polygon area according to Boustrophedon area path planning algorithm"""
        p = self.generate_path()
        return self.order_points(p, origin)

    def get_area_coverage(self, origin=None):
        if origin:
            origin = self.rotate_points(np.array([origin])).tolist()
        else:
            origin = self.rotate_points(np.array([self.origin])).tolist()
        result = self.boustrophedon_decomposition(origin)
        tf_result = self.rotate_from(np.array(result))
        return LineString(tf_result)


def vehicle_moniter():
    global vehicle1, vehicle2, vehicle3, vehicle4, vehicle5, vehicle6, vehicle7, vehicle8, vehicle9, vehicle10, vehicle11, vehicle12, vehicle13, vehicle14, vehicle15, vehicle16, vehicle17, vehicle18, vehicle19, vehicle20, vehicle21, vehicle22, vehicle23, vehicle24, vehicle25
    global count1,count2,count3,count4,count5,count6,count7,count8,count9,count10,count11,count12,count13,count14,count15,count16,count17,count18,count19,count20
    global count21,count22,count23,count24,count25
    global self_heal, master
    global self_heal_move_flag
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    count6 = 0
    count7 = 0
    count8 = 0 
    count9 = 0
    count10 = 0
    count11 = 0
    count12 = 0
    count13 = 0
    count14 = 0
    count15 = 0
    count16 = 0
    count17 = 0
    count18 = 0 
    count19 = 0
    count20 = 0
    count21 = 0
    count22 = 0
    count23 = 0 
    count24 = 0
    count25 = 0

    while True:                
        time.sleep(0.5)
        if count1 >= 1:        
            data1 = 5
        else:
            try:
                data1 = vehicle1.last_heartbeat
            except:
                data1=25

        if count2 >= 1:
            data2 = 5 
        else:
            try:
                data2 = vehicle2.last_heartbeat
            except:
                data2=25

        if count3 >= 1:
            data3 = 5 
        else:
            try:
                data3 = vehicle3.last_heartbeat
            except:
                data3=25
        if count4 >= 1:
            data4 = 5 
        else:
            try:
                data4 = vehicle4.last_heartbeat
            except:
                data4=25

        if count5 >= 1:
            data5 = 5 
        else:
            try:
                data5 = vehicle5.last_heartbeat
            except:
                data5=25

        if count6 >= 1:
            data6 = 5
        else:
            try:
                data6 = vehicle6.last_heartbeat
            except:
                data6=25

        if count7 >= 1:
            data7 = 5 
        else:
            try:
                data7 = vehicle7.last_heartbeat
            except:
                data7=25

        if count8 >= 1:
            data8 = 5 
            
        else:
            try:
                data8 = vehicle8.last_heartbeat
            except:
                data8=25

        if count9 >= 1:
           data9 = 5 
        else:
            try:
                data9 = vehicle9.last_heartbeat
            except:
                data9=25

        if count10 >= 1:
            data10 = 5 
        else:
            try:
                data10 = vehicle10.last_heartbeat

            except:
                data10=25
        if count11 >= 1:
            data11 = 5 
        else:
            try:
                data11 = vehicle11.last_heartbeat
            except:
                data11=25

        if count12 >= 1:
            data12 = 5 
        else:
            try:
                data12 = vehicle12.last_heartbeat
            except:
                data12=25

        if count13 >= 1:
                data13 = 5 
        else:
            try:
                data13 = vehicle13.last_heartbeat
            except:
                data13=25
        if count14 >= 1:
           data14 = 5 
        else:
            try:
                data14 = vehicle14.last_heartbeat
            except:
                data14=25

        if count15 >= 1:
           data15 = 5 
        else:
            try:
                data15 = vehicle15.last_heartbeat
            except:
                data15=25

        if count16 >= 1:
            data16 = 5 
        else:
            try:
                data16 = vehicle16.last_heartbeat
            except:
                data16=25

        if count17 >= 1:
            data17 = 5 
        else:
            try:
                data17 = vehicle17.last_heartbeat
            except:
                data17=25

        if count18 >= 1:
            data18 = 5 
        else:
            try:
                data18 = vehicle18.last_heartbeat              
            except:
                data18=25

        if count19 >= 1:
           data19 = 5 
        else:
            try:
                data19 = vehicle19.last_heartbeat

            except:
                data19=25

        if count20 >= 1:
            data20 = 5 
        else:
            try:
                data20 = vehicle20.last_heartbeat
            except:
                data20=25

        if count21 >= 1:
            data21 = 5 
        else:
            try:
                data21 = vehicle21.last_heartbeat
            except:
                data21=25

        if count22 >= 1:
            data22 = 5 
        else:
            try:
                data22 = vehicle22.last_heartbeat
            except:
                data22=25

        if count23 >= 1:
            data23 = 5 
        else:
            try:
                data23 = vehicle23.last_heartbeat               
            except:
                data23=25

        if count24 >= 1:
           data24 = 5 
        else:
            try:
                data24 = vehicle24.last_heartbeat

            except:
                data24=25

        if count25 >= 1:
            data25 = 5 
        else:
            try:
                data25 = vehicle25.last_heartbeat

            except:
                data25=25
            
        if data1 > 20:
            print "Err: Unable to connect to vehicle1."
            vehicle1 = None
            self_heal[0] = 1  
            data1_ = Label(text = "UAV_01", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 40)
            count1 = count1+1

	    if self_heal[0] == 1 and self_heal[1] != 2:		
		master = 2  #.............change of master 
	    elif (self_heal[0] == 1) and (self_heal[1] == 2) and (self_heal[2] != 3):		
		master = 3  #.............change of master 
	    elif (self_heal[0] == 1) and (self_heal[1] == 2) and (self_heal[2] == 3) and (self_heal[3] != 4):		
		master = 4  #.............change of master 
	    elif (self_heal[0] == 1) and (self_heal[1] == 2) and (self_heal[2] == 3) and (self_heal[3] == 4) and (self_heal[4] != 5):		
		master = 5  #.............change of master 

	    self_heal_move_flag = True
            """
	    time.sleep(15)
	    altitude()
	    time.sleep(15)
	    aggr()
	    time.sleep(10)
            """
        if data2 > 20:
            print "Err: Unable to connect to vehicle2."
            vehicle2 = None
            self_heal[1] = 2 
            data2_ = Label(text = "UAV_02", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 75)
            count2 = count2+1

	    if self_heal[0] == 1 and self_heal[1] != 2:		
		master = 2  #.............change of master 
	    elif (self_heal[0] == 1) and (self_heal[1] == 2) and (self_heal[2] != 3):		
		master = 3  #.............change of master 
	    elif (self_heal[0] == 1) and (self_heal[1] == 2) and (self_heal[2] == 3) and (self_heal[3] != 4):		
		master = 4  #.............change of master 
	    elif (self_heal[0] == 1) and (self_heal[1] == 2) and (self_heal[2] == 3) and (self_heal[3] == 4) and (self_heal[4] != 5):		
		master = 5  #.............change of master 
	    self_heal_move_flag = True
            """
	    time.sleep(15)
	    altitude()
	    time.sleep(15)
	    aggr()
	    time.sleep(10)
            """
        if data3 > 20:
            ###print "Err: Unable to connect to vehicle3."
            vehicle3 = None
            self_heal[2] = 3  
            data3_ = Label(text = "UAV_03", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 110)
            count3 = count3+1

	    if self_heal[0] == 1 and self_heal[1] != 2:		
		master = 2  #.............change of master 
	    elif (self_heal[0] == 1) and (self_heal[1] == 2) and (self_heal[2] != 3):		
		master = 3  #.............change of master 
	    elif (self_heal[0] == 1) and (self_heal[1] == 2) and (self_heal[2] == 3) and (self_heal[3] != 4):		
		master = 4  #.............change of master 
	    elif (self_heal[0] == 1) and (self_heal[1] == 2) and (self_heal[2] == 3) and (self_heal[3] == 4) and (self_heal[4] != 5):		
		master = 5  #.............change of master 
	    self_heal_move_flag = True
            """
	    time.sleep(15)
	    altitude()
	    time.sleep(15)
	    aggr()
	    time.sleep(10)
            """
        if data4 > 20:
            ###print "Err: Unable to connect to vehicle4."
            vehicle4 = None
            self_heal[3] = 4
            data4_ = Label(text = "UAV_04", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 145)
            count4 = count4+1

	    if self_heal[0] == 1 and self_heal[1] != 2:		
		master = 2  #.............change of master 
	    elif (self_heal[0] == 1) and (self_heal[1] == 2) and (self_heal[2] != 3):		
		master = 3  #.............change of master 
	    elif (self_heal[0] == 1) and (self_heal[1] == 2) and (self_heal[2] == 3) and (self_heal[3] != 4):		
		master = 4  #.............change of master 
	    elif (self_heal[0] == 1) and (self_heal[1] == 2) and (self_heal[2] == 3) and (self_heal[3] == 4) and (self_heal[4] != 5):		
		master = 5  #.............change of master 
	    self_heal_move_flag = True
            """
	    time.sleep(15)
	    altitude()
	    time.sleep(15)
	    aggr()
	    time.sleep(10)
            """
        if data5 > 20:
            ###print "Err: Unable to connect to vehicle5."
            vehicle5 = None
            self_heal[4] = 5
            data5_ = Label(text = "UAV_05", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 180)
            count5 = count5+1

	    if self_heal[0] == 1 and self_heal[1] != 2:		
		master = 2  #.............change of master 
	    elif (self_heal[0] == 1) and (self_heal[1] == 2) and (self_heal[2] != 3):		
		master = 3  #.............change of master 
	    elif (self_heal[0] == 1) and (self_heal[1] == 2) and (self_heal[2] == 3) and (self_heal[3] != 4):		
		master = 4  #.............change of master 
	    elif (self_heal[0] == 1) and (self_heal[1] == 2) and (self_heal[2] == 3) and (self_heal[3] == 4) and (self_heal[4] != 5):		
		master = 5  #.............change of master 
	    self_heal_move_flag = True
            """
	    time.sleep(15)
	    altitude()
	    time.sleep(15)
	    aggr()
	    time.sleep(10)
            """
        if data6 > 20:
            ###print "Err: Unable to connect to vehicle6."
            vehicle6 = None
            self_heal[5] = 6 
            data6_ = Label(text = "UAV_06", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 215)
            count6 = count6+1
	    self_heal_move_flag = True
            """
	    time.sleep(15)
	    altitude()
	    time.sleep(15)
	    aggr()
	    time.sleep(10)
            """
        if data7 > 20:
            ###print "Err: Unable to connect to vehicle7."
            vehicle7 = None
            self_heal[6] = 7 
            data7_ = Label(text = "UAV_07", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 250)
            count7 = count7+1
	    #self_heal_move_flag = True
            """
	    time.sleep(15)
	    altitude()
	    time.sleep(15)
	    aggr()
	    time.sleep(10)
            """
        if data8 > 20:
            ###print "Err: Unable to connect to vehicle8."
            vehicle8 = None
            self_heal[7] = 8
            data8_ = Label(text = "UAV_08", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 285)
            count8 = count8+1
	    #self_heal_move_flag = True
            """
	    time.sleep(15)
	    altitude()
	    time.sleep(15)
	    aggr()
	    time.sleep(10)
            """
        if data9 > 20:
            ###print "Err: Unable to connect to vehicle9."
            vehicle9 = None
            self_heal[8] = 9 
            data9_ = Label(text = "UAV_09", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 320)
            count9 = count9+1
	    #aggr()
        if data10 > 20:
            ###print "Err: Unable to connect to vehicle10."
            vehicle10 = None
            self_heal[9] = 10 
            data10_ = Label(text = "UAV_10", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 355)
            count10 = count10+1
	    #aggr()

        if data11 > 20:
            ####print "Err: Unable to connect to vehicle1."
            vehicle11 = None
            self_heal[10] = 1  
            data11_ = Label(text = "UAV_11", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 390)
            count11 = count11+1
	    #aggr()
        if data12 > 20:
            ###print "Err: Unable to connect to vehicle2."
            vehicle12 = None
            self_heal[11] = 1  
            data12_ = Label(text = "UAV_12", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 425)
            count12 = count12+1
	    #aggr()
        if data13 > 20:
            ###print "Err: Unable to connect to vehicle3."
            vehicle13 = None
            self_heal[12] = 1  
            data13_ = Label(text = "UAV_13", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 460)
            count13 = count13+1
	    #aggr()
        if data14 > 20:
            ###print "Err: Unable to connect to vehicle4."
            vehicle14 = None
            self_heal[13] = 1  
            data14_ = Label(text = "UAV_14", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 495)
            count14 = count14+1
	    #aggr()
        if data15 > 20:
            ###print "Err: Unable to connect to vehicle5."
            vehicle15 = None
            self_heal[14] = 1  
            data15_ = Label(text = "UAV_15", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 530)
            count15 = count15+1
	    #aggr()
	'''
        if data16 > 20:
            ###print "Err: Unable to connect to vehicle6."
            vehicle16 = None
            self_heal[15] = 1  
            data16_ = Label(text = "UAV_16", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 565)
            count16 = count16+1
	    #aggr()
        if data17 > 20:
            ###print "Err: Unable to connect to vehicle7."
            vehicle17 = None
            self_heal[16] = 1  
            data17_ = Label(text = "UAV_17", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 600)
            count17 = count17+1
	    #aggr()
        if data18 > 20:
            ###print "Err: Unable to connect to vehicle8."
            vehicle18 = None
            self_heal[17] = 1  
            data18_ = Label(text = "UAV_18", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 635)
            count18 = count18+1
	    #aggr()
        if data19 > 20:
            ###print "Err: Unable to connect to vehicle9."
            vehicle19 = None
            self_heal[18] = 1  
            data19_ = Label(text = "UAV_19", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 670)
            count19 = count19+1
	    #aggr()
        if data20 > 20:
            ###print "Err: Unable to connect to vehicle10."
            vehicle20 = None
            self_heal[19] = 1  
            data20_ = Label(text = "UAV_20", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 705)
            count20 = count20+1
	    #aggr()
        if data21 > 20:
            ###print "Err: Unable to connect to vehicle6."
            vehicle21 = None
            self_heal[20] = 1  
            data21_ = Label(text = "UAV_21", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 740)
            count21 = count21+1
	    #aggr()
        if data22 > 20:
            ###print "Err: Unable to connect to vehicle7."
            vehicle22 = None
            self_heal[21] = 1  
            data22_ = Label(text = "UAV_22", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 775)
            count22 = count22+1
	    #aggr()
        if data23 > 20:
            ###print "Err: Unable to connect to vehicle8."
            vehicle23 = None
            self_heal[22] = 1  
            data23_ = Label(text = "UAV_23", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 810)
            count23 = count23+1
	    #aggr()
        if data24 > 20:
            ###print "Err: Unable to connect to vehicle9."
            vehicle24 = None
            self_heal[23] = 1  
            data24_ = Label(text = "UAV_24", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 845)
            count24 = count24+1
	    #aggr()

        if data25 > 20:
            ###print "Err: Unable to connect to vehicle10."
            vehicle25 = None
            self_heal[24] = 1  
            data25_ = Label(text = "UAV_25", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 880)
            count25 = count25+1
	    #aggr()
	'''  
def receive_command_from_GCS():
    global self_heal  #move_all
    global master, no_uavs
    global xy_pos, latlon_pos
    global self_heal, self_heal
    global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10,vehicle11,vehicle12,vehicle13
    global vehicle14,vehicle15,vehicle16,vehicle17,vehicle18,vehicle19,vehicle20,vehicle21,vehicle22,vehicle23,vehicle24,vehicle25      
    global follower_host_tuple_G1,follower_host_tuple_G2,follower_host_tuple_G3,follower_host_tuple_G4,follower_host_tuple_G5
    global counter_G1,counter_G2,counter_G3,counter_G4,counter_G5
    global follower_host_tuple
    global circle_pos_flag
    xoffset = xoffset_entry.get()    
    cradius = cradius_entry.get()    
    aoffset = aoffset_entry.get()    
    salt = salt_entry.get() 
    xoffset = int(xoffset)    
    cradius = int(cradius)    
    aoffset = int(aoffset)
    salt = int(salt)
    print ("aggregation", master) 
    while True:
	    command_data, address = sock_rx1.recvfrom(1024)
	    cmd_data = command_data.split(',')
	    g_lat_cmd,g_lon_cmd = float(cmd_data[1]), float(cmd_data[2])
	    print ("..received move pos", g_lat_cmd,g_lon_cmd)
	    #..................all...................
	    if checkboxvalue_Group_all.get() == 1:
		    if master == 1:
			#lat = vehicle1.location.global_relative_frame.lat
			#lon = vehicle1.location.global_relative_frame.lon
			goto_lat=g_lat_cmd
			goto_lon=g_lon_cmd

			lat=float(goto_lat)
			lon=float(goto_lon)
		    if master == 2:
			#lon = vehicle2.location.global_relative_frame.lon
			##alt = vehicle2.location.global_relative_frame.alt
			#alt = 110

			goto_lat=g_lat_cmd
			goto_lon=g_lon_cmd

			lat=float(goto_lat)
			lon=float(goto_lon)

		    if master == 3:
			#lat = vehicle3.location.global_relative_frame.lat
			#lon = vehicle3.location.global_relative_frame.lon
			#alt = 120

			goto_lat=g_lat_cmd
			goto_lon=g_lon_cmd

			lat=float(goto_lat)
			lon=float(goto_lon)

		    if master == 4:
			#lat = vehicle4.location.global_relative_frame.lat
			#lon = vehicle4.location.global_relative_frame.lon
			#alt = 130

			goto_lat=g_lat_cmd
			goto_lon=g_lon_cmd

			lat=float(goto_lat)
			lon=float(goto_lon)

		    if master == 5:
			#lat = vehicle5.location.global_relative_frame.lat
			#lon = vehicle5.location.global_relative_frame.lon
			#alt = 140

			goto_lat=g_lat_cmd
			goto_lon=g_lon_cmd

			lat=float(goto_lat)
			lon=float(goto_lon)
		
		    if checkboxvalue1.get() == 1:
			formation(int(no_uavs), 'T', lat, lon)
		    elif checkboxvalue2.get() == 1:
			formation(int(no_uavs), 'L', lat, lon)
		    elif checkboxvalue3.get() == 1:
			formation(int(no_uavs), 'S', lat, lon)
		    elif checkboxvalue4.get() == 1:
			if circle_pos_flag == True:
			    circle_pos_flag = False
			    clat = clat_entry.get()
			    clon = clon_entry.get()
			    clat = float(clat)
			    clon = float(clon)
			    formation(int(no_uavs), 'C', clat, clon)
			else:
			    formation(int(no_uavs), 'C', lat, lon)


		    a,b,c = (0,0,0)
		    count_wp = 0
		    alt_000 = salt

		    for i, iter_follower in enumerate(follower_host_tuple): 
			#i = i+1
			if self_heal[i] > 0:
			    ##print "lost odd uav", self_heal[i]
			    print ("lost odd uav", (i+1))
			    pos_latlon = (0.0, 0.0)
			    latlon_pos.insert(i, (0.0, 0.0))
			    ##c = (c+20)   
			    if check_box_flag3 == True: 
				print ("self heal..to alt change")   
			    else:
				alt_000 = alt_000 + aoffset 
			else:   
			    test = latlon_pos[i]
			    print ("..t_goto..", test[0], test[1])        
			    target = LocationGlobalRelative(test[0], test[1], alt_000)
			    print ("target", target)
			    aggregation_formation(iter_follower,"GUIDED", target)        
			    alt_000 = alt_000 + aoffset
    
	    
def SERVER_send_gps_coordinate():
    global follower_host_tuple
    try:
        while True:
            for i, iter_follower in enumerate(follower_host_tuple):  
		if self_heal[i] == 0:  
			 if i == 0:
			 	filter_data1 = str('UAV1') + ',' + str(vehicle1.location.global_frame.lat) + ',' + str(vehicle1.location.global_frame.lon) + ',' + str(vehicle1.location.global_frame.alt) + ',' + str(vehicle1.groundspeed) + ',' + str(vehicle1.battery.level)+ ',' + str(vehicle1.mode.name)+ ',' + str(vehicle1.armed)+ ',' + str(vehicle1.attitude.roll)+ ',' + str(vehicle1.attitude.pitch)+ ',' + str(vehicle1.attitude.yaw)
			 if i == 1:
			 	filter_data1 = str('UAV2') + ',' + str(vehicle2.location.global_frame.lat) + ',' + str(vehicle2.location.global_frame.lon) + ',' + str(vehicle2.location.global_frame.alt) + ',' + str(vehicle2.groundspeed) + ',' + str(vehicle2.battery.level)+ ',' + str(vehicle2.mode.name)+ ',' + str(vehicle2.armed)+ ',' + str(vehicle2.attitude.roll)+ ',' + str(vehicle2.attitude.pitch)+ ',' + str(vehicle2.attitude.yaw)
			 if i == 2:
			 	filter_data1 = str('UAV3') + ',' + str(vehicle3.location.global_frame.lat) + ',' + str(vehicle3.location.global_frame.lon) + ',' + str(vehicle3.location.global_frame.alt) + ',' + str(vehicle3.groundspeed) + ',' + str(vehicle3.battery.level)+ ',' + str(vehicle3.mode.name)+ ',' + str(vehicle3.armed)+ ',' + str(vehicle3.attitude.roll)+ ',' + str(vehicle3.attitude.pitch)+ ',' + str(vehicle3.attitude.yaw)

			 if i == 3:
			 	filter_data1 = str('UAV4') + ',' + str(vehicle4.location.global_frame.lat) + ',' + str(vehicle4.location.global_frame.lon) + ',' + str(vehicle4.location.global_frame.alt) + ',' + str(vehicle4.groundspeed) + ',' + str(vehicle4.battery.level)+ ',' + str(vehicle4.mode.name)+ ',' + str(vehicle4.armed)+ ',' + str(vehicle4.attitude.roll)+ ',' + str(vehicle4.attitude.pitch)+ ',' + str(vehicle4.attitude.yaw)

			 if i == 4:
			 	filter_data1 = str('UAV5') + ',' + str(vehicle5.location.global_frame.lat) + ',' + str(vehicle5.location.global_frame.lon) + ',' + str(vehicle5.location.global_frame.alt) + ',' + str(vehicle5.groundspeed) + ',' + str(vehicle5.battery.level)+ ',' + str(vehicle5.mode.name)+ ',' + str(vehicle5.armed)+ ',' + str(vehicle5.attitude.roll)+ ',' + str(vehicle5.attitude.pitch)+ ',' + str(vehicle5.attitude.yaw)

			 if i == 5:
			 	filter_data1 = str('UAV6') + ',' + str(vehicle6.location.global_frame.lat) + ',' + str(vehicle6.location.global_frame.lon) + ',' + str(vehicle6.location.global_frame.alt) + ',' + str(vehicle6.groundspeed) + ',' + str(vehicle6.battery.level)+ ',' + str(vehicle6.mode.name)+ ',' + str(vehicle6.armed)+ ',' + str(vehicle6.attitude.roll)+ ',' + str(vehicle6.attitude.pitch)+ ',' + str(vehicle6.attitude.yaw)

			 if i == 6:
			 	filter_data1 = str('UAV7') + ',' + str(vehicle7.location.global_frame.lat) + ',' + str(vehicle7.location.global_frame.lon) + ',' + str(vehicle7.location.global_frame.alt) + ',' + str(vehicle7.groundspeed) + ',' + str(vehicle7.battery.level)+ ',' + str(vehicle7.mode.name)+ ',' + str(vehicle7.armed)+ ',' + str(vehicle7.attitude.roll)+ ',' + str(vehicle7.attitude.pitch)+ ',' + str(vehicle7.attitude.yaw)

			 if i == 7:
			 	filter_data1 = str('UAV8') + ',' + str(vehicle8.location.global_frame.lat) + ',' + str(vehicle8.location.global_frame.lon) + ',' + str(vehicle8.location.global_frame.alt) + ',' + str(vehicle8.groundspeed) + ',' + str(vehicle8.battery.level)+ ',' + str(vehicle8.mode.name)+ ',' + str(vehicle8.armed)+ ',' + str(vehicle8.attitude.roll)+ ',' + str(vehicle8.attitude.pitch)+ ',' + str(vehicle8.attitude.yaw)

			 if i == 8:
			 	filter_data1 = str('UAV9') + ',' + str(vehicle9.location.global_frame.lat) + ',' + str(vehicle9.location.global_frame.lon) + ',' + str(vehicle9.location.global_frame.alt) + ',' + str(vehicle9.groundspeed) + ',' + str(vehicle9.battery.level)+ ',' + str(vehicle9.mode.name)+ ',' + str(vehicle9.armed)+ ',' + str(vehicle9.attitude.roll)+ ',' + str(vehicle9.attitude.pitch)+ ',' + str(vehicle9.attitude.yaw)

			 if i == 9:
			 	filter_data1 = str('UAV10') + ',' + str(vehicle10.location.global_frame.lat) + ',' + str(vehicle10.location.global_frame.lon) + ',' + str(vehicle10.location.global_frame.alt) + ',' + str(vehicle10.groundspeed) + ',' + str(vehicle10.battery.level)+ ',' + str(vehicle10.mode.name)+ ',' + str(vehicle10.armed)+ ',' + str(vehicle10.attitude.roll)+ ',' + str(vehicle10.attitude.pitch)+ ',' + str(vehicle10.attitude.yaw)
			 #print (filter_data1)
			 sent = sock_tx1.sendto(filter_data1, server_address_tx1)
			 time.sleep(0.1)
		
    except:
	pass

# Check connections to router.
def CHECK_network_connection_odd(slave_host_odd, wait_time=None):
    global slave_odd_lost 
    global ch_e2, ch_e3, ch_e4, ch_e5
    print('{} - CHECK_network_connection({}) is started.'.format(time.ctime(), slave_host_odd))
    if wait_time == None:
        wait_time = 10 # Default wait time is 10 seconds.
    down_counter = 0
    while True:
        response = os.system('ping -c 1 ' + slave_host_odd)
        if response==0: # Link is OK.
            down_counter = 0 # Once connection is OK, reset counter.
            time.sleep(wait_time) # Check again in wait_time seconds.
            continue # Return back to the beginning of the while loop.
        else: # Link is down.
            down_counter += 1
            print('{} - Connection to router is DOWN for {} times.'.format(time.ctime(), down_counter))
            if down_counter > 2:
		if slave_host_odd == '192.168.6.210':
                	slave_odd_lost[0] = None
			ch_e3 = False
			aggr()  #....self heal uav lost 
		if slave_host_odd == '192.168.6.210':
                	slave_odd_lost[1] = None
			ch_e5 = False
			aggr()#....self heal uav lost 
		if slave_host_odd == '192.168.6.210':
                	slave_odd_lost[1] = None
			ch_e5 = False
			aggr()#....self heal uav lost 
		if slave_host_odd == '192.168.6.210':
                	slave_odd_lost[1] = None
			ch_e5 = False
			aggr()#....self heal uav lost 
                break 
            else: 
                print('{} - Check again in 1 seconds'.format(time.ctime()))
                time.sleep(1) # Check again in 2 seconds.

# Check connections to router.
def CHECK_network_connection_even(slave_host_even, wait_time=None):
    global slave_even_lost
    global ch_e2, ch_e3, ch_e4, ch_e5
    print('{} - CHECK_network_connection({}) is started.'.format(time.ctime(), slave_host_even))
    if wait_time == None:
        wait_time = 10 # Default wait time is 10 seconds.
    down_counter = 0
    while True:
        response = os.system('ping -c 1 ' + slave_host_even)
        if response==0: # Link is OK.
            down_counter = 0 # Once connection is OK, reset counter.
            time.sleep(wait_time) # Check again in wait_time seconds.
            continue # Return back to the beginning of the while loop.
        else: # Link is down.
            down_counter += 1
            print('{} - Connection to router is DOWN for {} times.'.format(time.ctime(), down_counter))
            if down_counter > 2:
		if slave_host_even == '192.168.6.210':
                	slave_even_lost[0] = None
			ch_e2 = False
			aggr()  #....self heal uav lost 
		if slave_host_even == '192.168.6.210':
                	slave_even_lost[1] = None
			ch_e4 = False
			aggr()  #....self heal uav lost 
		if slave_host_even == '192.168.6.210':
                	slave_even_lost[1] = None
			ch_e4 = False
			aggr()  #....self heal uav lost 
		if slave_host_even == '192.168.6.210':
                	slave_even_lost[1] = None
			ch_e4 = False
			aggr()  #....self heal uav lost 
		if slave_host_even == '192.168.6.210':
                	slave_even_lost[1] = None
			ch_e4 = False
			aggr()  #....self heal uav lost 
                break # Terminate while loop.
            else: # Have not reached max down times.
                print('{} - Check again in 1 seconds'.format(time.ctime()))
                time.sleep(1) # Check again in 2 seconds.


def CHECK_network_connection():
    global slave_heal_ip
    global follower_host_tuple
    for i,iter_follower in enumerate(follower_host_tuple):
        response = os.system('ping -c 1 ' + slave_heal_ip[i])
        if response==0: # Link is OK.
 	    print ("link is ok", slave_heal_ip[i])
        else: # Link is down.
 	    print ("link is down", slave_heal_ip[i])
 	    slave_heal_ip[i] = 'nolink'


def vehicle_connection():
    vehicle_count = 0
    global self_heal, self_heal
    global follower_host_tuple, heartbeat_timeout_data
    global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10,vehicle11,vehicle12,vehicle13
    global vehicle14,vehicle15,vehicle16,vehicle17,vehicle18,vehicle19,vehicle20,vehicle21,vehicle22,vehicle23,vehicle24,vehicle25 
    global follower_host_tuple_main, follower_host_tuple_sec,uavs_array
    ##no_of_uavs = no_of_uavs_entry.get()

    #print ("..........................no_of_uavs......", no_of_uavs)
    print ("............heartbeat_timeout_data........", heartbeat_timeout_data)
 
    try:
        #vehicle1 = connect('udpin:192.168.6.8:14551', baud=115200, heartbeat_timeout=heartbeat_timeout_data[0])
        vehicle1 = connect('udpin:192.168.6.210:14551', baud=115200, heartbeat_timeout=3)
        vehicle1.wait_ready('autopilot_version')
        time.sleep(0.1)
        vehicle1.airspeed = 10
        vehicle_count = vehicle_count + 1
        self_heal.append(0) 
        follower_host_tuple.append(vehicle1) 
        uavs_array.append('uav1')
        """
        lat1 = vehicle1.location.global_frame.lat
        lon1 = vehicle1.location.global_frame.lon

        = EleModel.GetElevation(lat1, lon1)
        print ("", )
        """
    except:
        print ("vehicle1 is lost")
        vehicle1 = None
        self_heal.append(1)  
    
    try:
        vehicle2 = connect('udpin:192.168.6.210:14552', baud=115200, heartbeat_timeout=3)
        vehicle2.wait_ready('autopilot_version')
        time.sleep(0.1)
        vehicle2.airspeed = 10
        vehicle_count = vehicle_count + 1
        self_heal.append(0)
        follower_host_tuple.append(vehicle2) 
	uavs_array.append('uav2')
        #
    except:
        print ("vehicle2 is lost")
        vehicle2= None
        self_heal.append(2)
        #time.sleep(0.5)
    try:
        vehicle3 = connect('udpin:192.168.6.210:14553', baud=115200, heartbeat_timeout=30)
        vehicle3.wait_ready('autopilot_version')
        time.sleep(0.1)
        vehicle3.airspeed = 10
        vehicle_count = vehicle_count + 1
        self_heal.append(0)   
        follower_host_tuple.append(vehicle3)  
	uavs_array.append('uav3')


    except:
        print ("vehicle3 is lost")
        vehicle3= None
        self_heal.append(3)        
        #time.sleep(0.5)

    try:
        vehicle4 = connect('udpin:192.168.6.210:14554', baud=115200, heartbeat_timeout=30)
        vehicle4.wait_ready('autopilot_version')
        time.sleep(0.1)
        vehicle4.airspeed = 10 
        vehicle_count = vehicle_count + 1
        self_heal.append(0)
        follower_host_tuple.append(vehicle4) 
        uavs_array.append('uav4')
        #
    except:
        print ("vehicle4 is lost")
        vehicle4 = None
        self_heal.append(4)
        #time.sleep(0.5)
    try:
        vehicle5 = connect('udpin:192.168.6.210:14555', baud=115200, heartbeat_timeout=30)
        vehicle5.wait_ready('autopilot_version')
        time.sleep(0.1)
        vehicle5.airspeed = 10
        vehicle_count = vehicle_count + 1
        self_heal.append(0)     
        follower_host_tuple.append(vehicle5)   
        uavs_array.append('uav5') 
        #
    except:
        print ("vehicle5 is lost")
        vehicle5 = None
        self_heal.append(5)        
        #time.sleep(0.5)
    try:
        vehicle6 = connect('udpin:192.168.6.210:14556', baud=115200, heartbeat_timeout=3)
        vehicle6.wait_ready('autopilot_version')
        time.sleep(0.1)
        vehicle6.airspeed = 10
        vehicle_count = vehicle_count + 1
        self_heal.append(0)
        follower_host_tuple.append(vehicle6) 
        uavs_array.append('uav6')
        #
    except:
        print ("vehicle6 is lost")
        vehicle6 = None
        self_heal.append(6)
        #time.sleep(0.5)
    try:
        vehicle7 = connect('udpin:192.168.6.210:14557', baud=115200, heartbeat_timeout=3)
        vehicle7.wait_ready('autopilot_version')
        time.sleep(0.1)
        vehicle7.airspeed = 10 
        vehicle_count = vehicle_count + 1
        self_heal.append(0) 
        follower_host_tuple.append(vehicle7)    
        uavs_array.append('uav7')    
        #
    except:
        print ("vehicle7 is lost")
        vehicle7 = None
        self_heal.append(7)        
        #time.sleep(0.5)
    try:
        vehicle8 = connect('udpin:192.168.6.210:14558', baud=115200, heartbeat_timeout=3)
        vehicle8.wait_ready('autopilot_version')
        time.sleep(0.1)
        vehicle8.airspeed = 10
        vehicle_count = vehicle_count + 1
        self_heal.append(0)
        follower_host_tuple.append(vehicle8) 
        uavs_array.append('uav8')
        #
    except:
        print ("vehicle8 is lost")
        vehicle8 = None
        self_heal.append(8)
        #time.sleep(0.5)
    try:
        vehicle9 = connect('udpin:192.168.6.210:14559', baud=115200, heartbeat_timeout=3)
        vehicle9.wait_ready('autopilot_version')
        time.sleep(0.1)
        vehicle9.airspeed = 10
        vehicle_count = vehicle_count + 1		
        self_heal.append(0)  
        follower_host_tuple.append(vehicle9)   
        uavs_array.append('uav9')    
        #
    except:
        print ("vehicle9 is lost")
        vehicle9 = None
        self_heal.append(9)        
        #time.sleep(0.5)
    try:
        vehicle10 = connect('udpin:192.168.6.210:14560', baud=115200, heartbeat_timeout=30)
        vehicle10.wait_ready('autopilot_version')
        time.sleep(0.1)
        vehicle10.airspeed = 10
        vehicle_count = vehicle_count + 1
        self_heal.append(0)
        follower_host_tuple.append(vehicle10) 
        uavs_array.append('uav10')
        #
    except:
        print ("vehicle10 is lost")
        vehicle10 = None
        self_heal.append(10)

    try:
        vehicle11 = connect('udpin:192.168.6.210:14561', baud=115200, heartbeat_timeout = 0.5)
        vehicle11.wait_ready('autopilot_version')
        time.sleep(0.1)
        vehicle11.airspeed = 10
        vehicle_count = vehicle_count + 1
        self_heal.append(0)  
        follower_host_tuple.append(vehicle11) 
        uavs_array.append('uav11')      
        #
    except:
        print ("vehicle11 is lost")
        vehicle11= None
        self_heal.append(11)        
        #time.sleep(0.5)
    try:
        vehicle12 = connect('udpin:192.168.6.210:14562', baud=115200, heartbeat_timeout = 0.5)
        vehicle12.wait_ready('autopilot_version')
        time.sleep(0.1)
        vehicle12.airspeed = 10
        vehicle_count = vehicle_count + 1
        self_heal.append(0)
        follower_host_tuple.append(vehicle12) 
        uavs_array.append('uav12')
        #
    except:
        print ("vehicle12 is lost")
        vehicle12= None
        self_heal.append(12)
        #time.sleep(0.5)

    try:
        vehicle13 = connect('udpin:192.168.6.210:14563', baud=115200, heartbeat_timeout = 0.5)
        vehicle13.wait_ready('autopilot_version')
        time.sleep(0.1)
        vehicle13.airspeed = 10 
        vehicle_count = vehicle_count + 1
        self_heal.append(0) 
        follower_host_tuple.append(vehicle13)        
        uavs_array.append('uav13')
    except:
        print ("vehicle13 is lost")
        vehicle13 = None
        self_heal.append(13)        
        #time.sleep(0.5)
    try:
        vehicle14 = connect('udpin:192.168.6.210:14564', baud=115200, heartbeat_timeout = 0.5)
        vehicle14.wait_ready('autopilot_version')
        time.sleep(0.1)
        vehicle14.airspeed = 10
        vehicle_count = vehicle_count + 1
        self_heal.append(0)
        follower_host_tuple.append(vehicle14) 
        uavs_array.append('uav14')
       
    except:
        print ("vehicle14 is lost")
        vehicle14 = None
        self_heal.append(14)
        #time.sleep(0.5)
    try:
        vehicle15 = connect('udpin:192.168.6.210:14565', baud=115200, heartbeat_timeout = 0.5)
        vehicle15.wait_ready('autopilot_version')
        time.sleep(0.1)
        vehicle15.airspeed = 10
        vehicle_count = vehicle_count + 1
        self_heal.append(0)  
        follower_host_tuple.append(vehicle15)       
        uavs_array.append('uav15')
    except:
        print ("vehicle15 is lost")
        vehicle15 = None
        self_heal.append(15) 
           
            
        #time.sleep(0.5)
    print("follower_host_tuple",follower_host_tuple)
    """
    try:
        vehicle16 = connect('udpin:192.168.6.210:14566', baud=115200, heartbeat_timeout = 0.5)
        vehicle16.wait_ready('autopilot_version')
        time.sleep(0.1)
        vehicle16.airspeed = 10 
        vehicle_count = vehicle_count + 1
        self_heal.append(0)
        #
    except:
        print ("vehicle16 is lost")
        vehicle16 = None
        self_heal.append(16)
        #time.sleep(0.5)
    try:
        vehicle17 = connect('udpin:192.168.6.210:14567', baud=115200, heartbeat_timeout = 0.5)
        vehicle17.wait_ready('autopilot_version')
        time.sleep(0.1)
        vehicle17.airspeed = 10
        vehicle_count = vehicle_count + 1
        self_heal.append(0)        
        #
    except:
        print ("vehicle17 is lost")
        vehicle17 = None
        self_heal.append(17)        
        #time.sleep(0.5)
    try:
        vehicle18 = connect('udpin:192.168.6.210:14568', baud=115200, heartbeat_timeout = 0.5)
        vehicle18.wait_ready('autopilot_version')
        time.sleep(0.1)
        vehicle18.airspeed = 10
        vehicle_count = vehicle_count + 1
        self_heal.append(0)
        #
    except:
        print ("vehicle18 is lost")
        vehicle18 = None
        self_heal.append(18)
        #time.sleep(0.5)
    try:
        vehicle19 = connect('udpin:192.168.6.210:14569', baud=115200, heartbeat_timeout = 0.5)
        vehicle19.wait_ready('autopilot_version')
        time.sleep(0.1)
        vehicle19.airspeed = 10
        vehicle_count = vehicle_count + 1
        self_heal.append(0)        
        #
    except:
        print ("vehicle19 is lost")
        vehicle19 = None
        self_heal.append(19)        

    try:
        vehicle20 = connect('udpin:192.168.6.210:14570', baud=115200, heartbeat_timeout = 0.5)
        vehicle20.wait_ready('autopilot_version')
        time.sleep(0.1)
        vehicle20.airspeed = 10
        vehicle_count = vehicle_count + 1
        self_heal.append(0)
        #
    except:
        print ("vehicle20 is lost")
        vehicle20 = None
        self_heal.append(20)

    try:
        vehicle21 = connect('udpin:192.168.6.210:14571', baud=115200, heartbeat_timeout = 0.5)
        vehicle21.wait_ready('autopilot_version')
        time.sleep(0.1)
        vehicle21.airspeed = 10 
        vehicle_count = vehicle_count + 1
        self_heal.append(0)        
        #
    except:
        print ("vehicle21 is lost")
        vehicle21 = None
        self_heal.append(21)        
        #time.sleep(0.5)
    try:
        vehicle22 = connect('udpin:192.168.6.210:14572', baud=115200, heartbeat_timeout = 0.5)
        vehicle22.wait_ready('autopilot_version')
        time.sleep(0.1)
        vehicle22.airspeed = 10
        vehicle_count = vehicle_count + 1
        self_heal.append(0)
        #
    except:
        print ("vehicle22 is lost")
        vehicle22 = None
        self_heal.append(22)
        #time.sleep(0.5)
    try:
        vehicle23 = connect('udpin:192.168.6.210:14573', baud=115200, heartbeat_timeout = 0.5)
        vehicle23.wait_ready('autopilot_version')
        time.sleep(0.1)
        vehicle23.airspeed = 10
        vehicle_count = vehicle_count + 1
        self_heal.append(0)        
        #
    except:
        print ("vehicle23 is lost")
        vehicle23 = None
        self_heal.append(23)        
        #time.sleep(0.5)
    try:
        vehicle24 = connect('udpin:192.168.6.210:14574', baud=115200, heartbeat_timeout = 0.5)
        vehicle24.wait_ready('autopilot_version')
        time.sleep(0.1)
        vehicle24.airspeed = 10
        vehicle_count = vehicle_count + 1
        self_heal.append(0)
        #
    except:
        print ("vehicle24 is lost")
        vehicle24 = None
        self_heal.append(24)

    try:
        vehicle25 = connect('udpin:192.168.6.210:14575', baud=115200, heartbeat_timeout = 0.5)
        vehicle25.wait_ready('autopilot_version')
        time.sleep(0.1)
        vehicle25.airspeed = 10
        vehicle_count = vehicle_count + 1
        self_heal.append(0)        
        #
    except:
        print ("vehicle25 is lost")
        vehicle25 = None
        self_heal.append(25)        
    """
    #follower_host_tuple =[vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10,vehicle11,vehicle12,vehicle13,vehicle14,vehicle15]
    follower_host_tuple_main =[vehicle1, vehicle2, vehicle3, vehicle4, vehicle5]
    follower_host_tuple_sec =[vehicle6,vehicle7,vehicle8,vehicle9,vehicle10,vehicle11,vehicle12,vehicle13,vehicle14,vehicle15]

    print ("all uav connection check complete")
    for i, iter_follower in enumerate(follower_host_tuple):     
        home_loc(iter_follower)

    
    print ("home_location:", home_location)
    t3 = threading.Thread(target = vehicle_moniter)
    t3.daemon = True
    t3.start() 
    
    """
    a2 = threading.Thread(target = vehicle_RTH_moniter)
    a2.daemon = True
    a2.start()
    """
    """
    a2_ = threading.Thread(target = self_heal_adjust)
    a2_.daemon = True
    a2_.start()
    """
    '''
    a3 = threading.Thread(target = move_all_pos_guided)
    a3.daemon = True
    a3.start() 

    a4 = threading.Thread(target = move_all_pos_guided_return)
    a4.daemon = True
    a4.start() 
    
    a6 = threading.Thread(target = SERVER_send_gps_coordinate)
    a6.daemon = True
    a6.start()
    
    a7 = threading.Thread(target = receive_command_from_GCS)
    a7.daemon = True
    a7.start()  
    '''
    #.....CHECK_network_connection()
    print ("............####################....................")
    #....print ("...slave_heal_ip...", slave_heal_ip)

    """
    for iter_follower_odd in follower_host_tuple_odd:
	threading.Thread(target=CHECK_network_connection_odd,args=(iter_follower_odd,),kwargs={'wait_time':10}).start()
	
    for iter_follower_even in follower_host_tuple_even:
	threading.Thread(target=CHECK_network_connection_even,args=(iter_follower_even,),kwargs={'wait_time':10}).start()
    """

    

def home_loc(vehicle):
    global home_location
    if vehicle == None:
        print ("slave is lost")
        home_loc = (13.3861723, 80.2331543)
        home_location.append(home_loc)
    else:
        h_lat = vehicle.location.global_frame.lat
        h_lon = vehicle.location.global_frame.lon
        home_loc = (h_lat, h_lon)
        print (home_loc)
        print (home_location)
        home_location.append(home_loc)

export_mission_filename = 'exportedmission.txt'
count = 0

def update_ip_link_status():
    global follower_host_tuple_ip, ip_status, heartbeat_timeout_data

    slave01 = '127.0.0.1'  #vehicle01
    slave02 = '127.0.0.1' #vehicle02
    slave03 = '127.0.0.1'
    slave04 = '127.0.0.1'
    slave05 = '127.0.0.1'
    slave06 = '127.0.0.1'
    slave07 = '127.0.0.1'
    slave08 = '127.0.0.1'
    slave09 = '127.0.0.1'
    slave10 = '127.0.0.1'
    slave11 = '127.0.0.1'  #vehicle01
    slave12 = '127.0.0.1' #vehicle02
    slave13 = '127.0.0.1'
    slave14 = '127.0.0.1'
    slave15 = '127.0.0.1'
    slave16 = '127.0.0.1'
    slave17 = '127.0.0.1'
    slave18 = '127.0.0.1'
    slave19 = '127.0.0.1'
    slave20 = '127.0.0.1'
    slave21 = '127.0.0.1'
    slave22 = '127.0.0.1'
    slave23 = '127.0.0.1'
    slave24 = '127.0.0.1'
    slave25 = '127.0.0.1'


    follower_host_tuple_ip = [slave01, slave02, slave03, slave04, slave05, slave06, slave07, slave08, slave09, slave10, slave11, slave12, slave13, slave14, slave15, slave16, slave17, slave18, slave19, slave20, slave21, slave22, slave23, slave24, slave25]

    ##while True:
    for i,follower_host in enumerate(follower_host_tuple_ip):
        response = os.system('ping -c 1 ' + follower_host)
        if response==0: # Link is OK.
                print ("link ok:", follower_host)
                ip_status[i] = True
                heartbeat_timeout_data.append(30)
        else:
                print ("no_link:", follower_host)
                iter_follower_status = False
                ip_status[i] = False
                heartbeat_timeout_data.append(2)

def rc_connection():
    global follower_host_tuple
    while True:
        for iter_follower in follower_host_tuple:  
            if iter_follower == None:
                dat123 = 0
            else:        
                iter_follower.channels.overrides['1'] = 1700
                iter_follower.channels.overrides['2'] = 1700
                iter_follower.channels.overrides['3'] = 1700
                iter_follower.channels.overrides['4'] = 1700
                iter_follower.channels.overrides['5'] = 1700
                iter_follower.channels.overrides['6'] = 1700
        time.sleep(0.5)

def self_heal_adjust():
    global takeoff_flag, RTH_flag
    global follower_host_tuple, RTL_all_flag, master
    global count1,count2,count3,count4,count5,count6,count7,count8,count9,count10
    global self_heal_move_flag, self_heal
    global control_command
    while True:
	time.sleep(1)
	if self_heal_move_flag == True:
		if control_command == True:
			print (".....self_heal_move_flag == True.....")
			print ("one uav rtl or lost")
			print ("...self_heal..", self_heal)
			for i, iter_follower in enumerate(follower_host_tuple): 
				if self_heal[i] > 0:
					print ("lost odd uav", (i+1))
				else:
					print ("present uav")  
					for i in range(0, 5):
						if iter_follower.mode.name =="RTL":
							dataf = 10
						else:
							threading.Thread(target=air_break,args=(iter_follower,)).start()
							#air_break(iter_follower)
							time.sleep(0.2)
			#..............RTL...............
			time.sleep(5)
                        for i in range(0, 2):
				altitude()
			time.sleep(8)
			aggr()
			time.sleep(10)
			#..................................
			self_heal_move_flag = False

def vehicle_RTH_moniter():
    global takeoff_flag, RTH_flag
    global follower_host_tuple, RTL_all_flag, master
    global count1,count2,count3,count4,count5,count6,count7,count8,count9,count10
    global self_heal_move_flag, self_heal
    global RTH_array
    #RTH_array = [100,100,100,100,100,100,100,100]
    pause()
    time.sleep(5)
    while True:
        for i,iter_follower in enumerate(follower_host_tuple):
            if self_heal[i] > 0:
                dat123=50
            else:  
		if RTL_all_flag == True:
			hel = 1
			#RTH_array = [100,100,100,100,100,100,100,100]
			
			self_heal_move_flag = False
		else:   
		        if iter_follower.mode.name =="RTL":
				self_heal[i] = i+1
				if self_heal[0] == 1 and self_heal[1] != 2:		
					master = 2  #.............change of master 
				elif (self_heal[0] == 1) and (self_heal[1] == 2) and (self_heal[2] != 3):		
					master = 3  #.............change of master 
				elif (self_heal[0] == 1) and (self_heal[1] == 2) and (self_heal[2] == 3) and (self_heal[3] != 4):		
					master = 4  #.............change of master 
				elif (self_heal[0] == 1) and (self_heal[1] == 2) and (self_heal[2] == 3) and (self_heal[3] == 4) and (self_heal[4] != 5):		
					master = 5  #.............change of master 

				if i == RTH_array[i]:
					heloo = 1
				else:
					print ("..Rth...aggr...and time.sleep start....")
					print ("....master....", master)
					self_heal_move_flag = True

					#RTH_flag = True
					"""
					if takeoff_flag == True:
						print ("takeoff_flag is enable")
					if wp_navigation_flag = False:

					else:
						time.sleep(15)
						altitude()
						time.sleep(15)
						aggr()
						time.sleep(10)
					"""

				RTH_array[i] = i
        time.sleep(1)



def autonumus_connection_1():
    global takeoff_flag, search_flag
    global control_command, vehicle1, count_123, wp_navigation_flag, wp_navigation_return_flag, aggr_and_rtl_flag
    count_123 = 0
    while True:
        time.sleep(0.5)
        if (control_command) and (count_123 == 0):
	    count_123 = count_123 + 1
            takeoff_flag = True  # takeoff_flag
            #.#####################......tttt.......
            """
	    print ("...autonumus...takeoff_all....")
	    for i in range(0, 2):
		    takeoff_all()
		    time.sleep(4)

	    print ("...takeoff sleep 10sec....")
	    time.sleep(10)
            #........abort mission.........
	    if control_command == False:
	    	print ("autonumus stop")
		count_123 = 0
	    	break
	    #..............................
	    for i in range(0, 2):
		    print ("altitude increase function")
		    altitude()
                    time.sleep(2)
	    time.sleep(200) 
            speed()
            """
            #################........ttt..........
            
	    takeoff_flag = False
	    print ("...autonumus...altitude increase function...60sec.")
	    print ("...autonumus...aggr.....")
	    #........abort mission.........
	    if control_command == False:
	    	print ("autonumus stop")
		count_123 = 0
	    	break
	    #.............................
	    #aggr()
            #.........abort mission.........
	    if control_command == False:
	    	print ("autonumus stop")
		count_123 = 0
	    	break
	    #.............................
	    #time.sleep(2)

	    speed()
	    time.sleep(2)	    
	    print ("...autonumus...move.wp..to target..p1,p2,p3")
	    wp_navigation_flag = True	    
        if (control_command) and (aggr_and_rtl_flag == True):
		aggr_and_rtl_flag = False
		print ("...target search timer start...150 sec..")
                #................................
		count_swarm = 0
		for i in range(0, len(self_heal)):
			if (int(self_heal[i]) > 0):
				print ("lost uav for swarm")
			else:
				count_swarm = count_swarm+1
		print ("....no of uav avilable for swarm", count_swarm)

                if count_swarm == 8:
                	#...........timeer for search.....
			for m in range(0, 180): 
				print ("8-UAV search mode....100s")
				search_flag = True
				time.sleep(1)  #......time for uav Target serach 
		    		#.........abort mission.........
				if control_command == False:
			    		print ("autonumus stop")
			    		count_123 = 0
					break
			#pause()

                if count_swarm == 7:
                	#...........timeer for search.....
			for m in range(0, 190): 
				print ("7-UAV search mode....120s")
				search_flag = True
				time.sleep(1)  #......time for uav Target serach 
		    		#.........abort mission.........
				if control_command == False:
			    		print ("autonumus stop")
			    		count_123 = 0
					break
			#pause()

                if count_swarm == 6:
                	#...........timeer for search.....
			for m in range(0, 190): 
				print ("6-UAV search mode...140s")
				search_flag = True
				time.sleep(1)  #......time for uav Target serach 
		    		#.........abort mission.........
				if control_command == False:
			    		print ("autonumus stop")
			    		count_123 = 0
					break
			#pause()

                if count_swarm == 5:
                	#...........timeer for search.....
			for m in range(0, 200): 
				print ("5-UAV search mode....160s")
				search_flag = True
				time.sleep(1)  #......time for uav Target serach 
		    		#.........abort mission.........
				if control_command == False:
			    		print ("autonumus stop")
			    		count_123 = 0
					break
			#pause()

                if count_swarm == 4:
                	#...........timeer for search.....
			for m in range(0, 200): 
				print ("4-UAV search mode....180s")
				search_flag = True
				time.sleep(1)  #......time for uav Target serach 
		    		#.........abort mission.........
				if control_command == False:
			    		print ("autonumus stop")
			    		count_123 = 0
					break
			#pause()
                if count_swarm == 3:
                	#...........timeer for search.....
			for m in range(0, 230): 
				print ("3-UAV search mode....200s")
				search_flag = True
				time.sleep(1)  #......time for uav Target serach 
		    		#.........abort mission.........
				if control_command == False:
			    		print ("autonumus stop")
			    		count_123 = 0
					break
			#pause()

                if count_swarm == 2:
                	#...........timeer for search.....
			for m in range(0, 250): 
				print ("2-UAV search mode....220s")
				search_flag = True
				time.sleep(1)  #......time for uav Target serach 
		    		#.........abort mission.........
				if control_command == False:
			    		print ("autonumus stop")
			    		count_123 = 0
					break
			#pause()

                if count_swarm == 1:
                	#...........timeer for search.....
			for m in range(0,300): 
				print ("1-UAV search mode....190s")
				search_flag = True
				time.sleep(1)  #......time for uav Target serach 
		    		#.........abort mission.........
				if control_command == False:
			    		print ("autonumus stop")
			    		count_123 = 0
					break
			#pause()

		#................................
		print ("UAV search mode....>>>>>>>>>>>>>>>>>>>>>>>>>>...........complete")
		search_flag = False
                for i in range(0, 2):	
			altitude()   #...altitude increase to original height
		time.sleep(20)
		#.............abort mission...........
		if control_command == False:
		    	print ("autonumus stop")
		    	count_123 = 0
			break
                #.................................
		speed()
		aggr()
		time.sleep(10)
		print ("all UAV aggr to 30 sec")
		time.sleep(15) #.....aggr to all uav
		#........abort mission..............
		if control_command == False:
			print ("autonumus stop")
	    		count_123 = 0
			break
                #..................................
		speed()
		print ("...autonumus...move.return wp to home..p7,p6,p5..home")
		wp_navigation_return_flag = True

def receive_target_image():
	while True:
		(rpiName, frame) = imageHub.recv_image()
		print ("rpiName", rpiName)
		data = rpiName.split(',')
		imageHub.send_reply(b'OK')
		if data[0] == 'UAV5': 
			frame1 = frame
			path = str(data[0])+','+str(data[1])+','+str(data[2])+','+str(data[3])+','+str(data[4])+','+str(data[5])+','+str(data[6])

			cv2.imwrite(path_01+path+'.jpg', frame1)
	    		#list_of_files = glob.glob('./jeo_tag/*')
			#list_of_files = glob.glob('./jeo_tag/*')
		
			#latest_file = max(list_of_files, key=os.path.getctime)
			#set_gps_location(path_01+path+'.jpg', float(data[2]), float(data[3]), float(data[4]))
		if data[0] == 'UAV2': 
			frame2 = frame
			path = str(data[0])+','+str(data[1])+','+str(data[2])+','+str(data[3])+','+str(data[4])+','+str(data[5])+','+str(data[6])
			cv2.imwrite(path_01+path+'.jpg', frame2)
		if data[0] == 'UAV3': 
			frame3 = frame
			path = str(data[0])+','+str(data[1])+','+str(data[2])+','+str(data[3])+','+str(data[4])+','+str(data[5])+','+str(data[6]) 
			cv2.imwrite(path_01+path+'.jpg', frame3)
		if data[0] == 'UAV4': 
			frame4 = frame
			path = str(data[0])+','+str(data[1])+','+str(data[2])+','+str(data[3])+','+str(data[4])+','+str(data[5])+','+str(data[6])
			cv2.imwrite(path_01+path+'.jpg', frame4)
		if data[0] == 'UAV1': 
			frame5 = frame
			path = str(data[0])+','+str(data[1])+','+str(data[2])+','+str(data[3])+','+str(data[4])+','+str(data[5])+','+str(data[6])
			cv2.imwrite(path_01+path+'.jpg', frame5)

	    

def autonumus_connection():
    global control_command, vehicle1
    xoffset = int(10)
    cradius = int(10)
    flightLevel = 25.0

    counter = 2

    flightLevel = 25.0
    minDistance = 30.0
    collisionFlag = True
    robotRadius = 10.0
    numDrones = 9
    qq = []
    """
    #gotoAlt_arry = [vehicle1.location.global_relative_frame.alt, (vehicle1.location.global_relative_frame.alt)+3, (vehicle1.location.global_relative_frame.alt)+6, (vehicle1.location.global_relative_frame.alt)+9, (vehicle1.location.global_relative_frame.alt)+12, (vehicle1.location.global_relative_frame.alt)+15, (vehicle1.location.global_relative_frame.alt)+18, (vehicle1.location.global_relative_frame.alt)+21, (vehicle1.location.global_relative_frame.alt)+24, (vehicle1.location.global_relative_frame.alt)+27, (vehicle1.location.global_relative_frame.alt)+30]
    while True:
        time.sleep(0.5)

        if control_command:
            gotoAlt_arry = [vehicle1.location.global_relative_frame.alt, (vehicle1.location.global_relative_frame.alt)+3, (vehicle1.location.global_relative_frame.alt)+6, (vehicle1.location.global_relative_frame.alt)+9, (vehicle1.location.global_relative_frame.alt)+12, (vehicle1.location.global_relative_frame.alt)+15, (vehicle1.location.global_relative_frame.alt)+18, (vehicle1.location.global_relative_frame.alt)+21, (vehicle1.location.global_relative_frame.alt)+24, (vehicle1.location.global_relative_frame.alt)+27, (vehicle1.location.global_relative_frame.alt)+30, (vehicle1.location.global_relative_frame.alt)+33]
            print ("..............loop...collision...................")
            loc = []
            flag = False
            
            #rectangle = pt.grid(numDrones=numDrones, distNodes=20, origin=(loadVehicles[0].location.global_frame.lat, loadVehicles[0].location.global_frame.lon))
            #triangle = pt.triangle(numDrones=numDrones, distNodes=20, origin=(loadVehicles[0].location.global_frame.lat, loadVehicles[0].location.global_frame.lon))
            #line = pt.line(numDrones=numDrones, distNodes=20, origin=(loadVehicles[0].location.global_frame.lat, loadVehicles[0].location.global_frame.lon), lineType='H')
            
            if counter == 1:
                rectangle = pt.grid(numDrones=numDrones, distNodes=15, origin=(vehicle1.location.global_frame.lat, vehicle1.location.global_frame.lon))
                dest = rectangle
            elif counter == 2:
                triangle = pt.triangle(numDrones=numDrones, distNodes=15, origin=(vehicle1.location.global_frame.lat, vehicle1.location.global_frame.lon))
                dest = triangle
            elif counter == 3:
                line = pt.line(numDrones=numDrones, distNodes=15, origin=(vehicle1.location.global_frame.lat, vehicle1.location.global_frame.lon), lineType='H')
                dest = line


            for val in range(0, numDrones):
                exec(f"location{val} = (follower_host_tuple[val].location.global_frame.lat, follower_host_tuple[val].location.global_frame.lon)")
                exec(f"loc.append(location{val})")

            for dests in range(1, len(dest)+1):
                exec(f"dest{dests} = [((dest[dests-1][0]),(dest[dests-1][1]))]")

            for current in range(1, len(loc)+1):
                obstacle_list = []
                obstaclesXY = []
            # print("current", current)
            for each in range(1, len(loc)+1):
                if current != each:
                    distance, bearing = locate.distance_bearing(loc[current-1][0], loc[current-1][1], loc[each-1][0], loc[each-1][1])
                    obstacle_list.append(loc[each-1])

            # Converting current vehicle coordinates, destination and other vehicle coordinates to x,y coordinates 
            sourceX, sourceY = locate.geoToCart(origin=loc[0], endDistance=1000, geoLocation=(loc[current-1][0],loc[current-1][1]))
            destX, destY = locate.geoToCart(origin=loc[0], endDistance=1000, geoLocation=(dest[current-1][0],dest[current-1][1]))
            for obstacles in obstacle_list:
                obsX, obsY = locate.geoToCart(origin=loc[0], endDistance=1000, geoLocation=(obstacles[0],obstacles[1]))
                obstaclesXY.append((float(obsX), float(obsY), float(robotRadius)))

            # Converting All the Inputs into float
            source, destination = [float(sourceX), float(sourceY)], [float(destX), float(destY)]

            if [abs(math.floor(source[0])),abs(math.floor(source[1]))] != [abs(math.floor(destination[0])),abs(math.floor(destination[1]))]:
                # Determing Vehicle Path
                print("vehicle", current)

                # Planning Path using RRT*
                rrt_star = rsc.RRTStar(
                start=source,
                goal=destination,
                rand_area=[-200, 1000],
                obstacle_list=obstaclesXY,
                expand_dis=2.0)

                # RRT* determined path
                path = rrt_star.planning()

                if path is None:
                    #print("Cannot find path")
                    dat = 2

                else:
                    path = path[::-1]
                    lenCurrent = len(eval('dest' + str(current)))

                    if lenCurrent > 0:
                        exec(f"dest{current} = dest{current}[-1:]")

                    if len(path) >= 1:
                        pathLat, pathLon = locate.cartToGeo(origin = loc[0], endDistance = 1000, cartLocation = (path[1][0], path[1][1]))
                    else:
                        pathLat, pathLon = locate.cartToGeo(origin = loc[0], endDistance = 1000, cartLocation = (path[0][0], path[0][1]))
                    exec(f"dest{current}.insert(0,(float(pathLat), float(pathLon)))")
                    # print("dest", current, eval("dest"+str(current)))
            if distance < 10 :
                qq.append(distance)
                #print("distance", distance)
                print(qq)
                print("vehicle", current)
                # plt.show()
            for dests in range(1, len(dest)+1):
                for val in eval("dest"+str(dests)):
                    gotoLat, gotoLon, gotoAlt = val[0],val[1], float(flightLevel)
                    print ("vehicle simple_goto", dests-1)
                    if int(dests-1) == 0:
                            print ("uav_1")
                    else:
                        follower_host_tuple[dests-1].simple_goto(LocationGlobalRelative(gotoLat, gotoLon, gotoAlt_arry[dests-1]))
                        time.sleep(1)

            try:
                xoffset = xoffset_entry.get()
                cradius = cradius_entry.get()
                xoffset = int(xoffset)
                cradius = int(cradius)
                if xoffset == 10 and cradius == 10:
                    counter = 2
                elif xoffset == 20 and cradius == 20:
                    counter = 1
                elif xoffset == 30 and cradius == 30:
                    counter = 3
            except:
                pass

"""
                     
def vehicle_connect():
    t2 = threading.Thread(target = update_gui)
    t2.daemon = True
    t2.start()

    time.sleep(2)

    update_ip_link_status()

    """autonumus_connection_1
    t3 = threading.Thread(target = update_ip_link_status)
    t3.daemon = True
    t3.start() 
    """

    t1 = threading.Thread(target=vehicle_connection)
    t1.daemon = True
    t1.start()
    
    a1 = threading.Thread(target=autonumus_connection_1)
    a1.daemon = True
    a1.start()
    """
    a_123 = threading.Thread(target=receive_target_image)
    a_123.daemon = True
    a_123.start()
    """


    #a2 = threading.Thread(target=rc_connection)
    #a2.daemon = True
    #a2.start()
# Create a socket for receiving data
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('', 12009)  # receive from .....rx.py
sock.bind(server_address)
sock.setblocking(0)

# Load obstacles from YAML file
obstacles = []
def load_yaml(filename):
    global obstacles
    obstacles=[]
    
    with open(filename, 'r') as f:
        dict_ = yaml.safe_load(f)
        size = (dict_['size']['x'], dict_['size']['y'])
        if dict_['obstacles'] is not None:
            obstacle_list = dict_['obstacles']
            for o in obstacle_list:
                obstacles.append(Polygon(o))
        name = dict_['name']
        print("Loaded world: " + name)
        return size, obstacles



def update_load_yaml(filename):
    global obstacles
    obstacles=[]
    with open(filename, 'r') as f:
        dict_ = yaml.safe_load(f)
        size = (dict_['size']['x'], dict_['size']['y'])
        if dict_['obstacles'] is not None:
            obstacle_list = dict_['obstacles']
            for o in obstacle_list:
                obstacles.append(Polygon(o))
        name = dict_['name']
        print("Loaded world: " + name)
        return size, obstacles

def open_plot():
    
    print("plot opened...........")
    global frame_4,fileloc,filename,filepath
    plt.close('all')
    #fig.clear()
    # Load data from YAML file
    print("fileloc",fileloc,filename)
    
    filepath=fileloc+filename+".yaml"
    size, obstacles = load_yaml(filepath	)
    
    print(size, "size")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('UAV Positions')
    ax.clear()
    # Plot obstacles
    for obs in obstacles:
        x, y = obs.exterior.xy
        ax.fill(x, y, fc='gray', alpha=0.9)

    # Set limits
    ax.set_xlim(0, size[0])
    ax.set_ylim(0, size[1])
    
    
    if frame_4 is None:
        # Plot the graph directly
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().place(x=1000, y=30, width=900, height=800)
        
    else:
    
        if hasattr(frame_4, 'canvas'):
            frame_4.canvas.get_tk_widget().destroy()  # Clear the canvas
        else:
            frame_4.canvas = FigureCanvasTkAgg(fig, master=frame_4)
        frame_4.canvas.draw_idle()
        frame_4.canvas.get_tk_widget().place(x=1000, y=30, width=900, height=800)
    #update_plot()
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    # Place the toolbar in your Tkinter window
    toolbar.place(x=1300, y=870, width=900, height=50)

"""
def open_plot():
	fig, ax = plt.subplots(figsize=(10, 10))
	size,obs=load_yaml(filename="/home/dhaksha/Documents/socket/dce_swarm_nav/swarm_tasks-main/swarm_tasks/Examples/basic_tasks/rectangles_cam.yaml")
        print(size,"size")
	try:	    
	    while True:  
		for obs in obstacles:
			x,y = obs.exterior.xy
			plt.fill(x,y, fc='gray', alpha=0.9)

		    #plt.draw()
		plt.title('UAV Positions')
		    #plt.text(x[-1], y[-1],f"Elapsed Time: {elapsed_time} seconds", ha='right', va='top')
		plt.xlabel('X-axis')
		plt.ylabel('Y-axis')
	       # plt.legend()

		plt.xlim(0, size[0])
		plt.ylim(0, size[1])
		plt.pause(0.05)
	    plt.show()
	except KeyboardInterrupt:
	    print("Interrupted by user")

"""
return_flag=False
fig=plt.Figure(figsize=(9,9))
ax=fig.add_subplot(111)
home_pos=[]
goal_points=[]
goal_table=[]
return_goal_table=[]
home_pos_flag=False
goal_points_flag=False
uav_colors = OrderedDict([
    ('uav1', 'b'),
    ('uav2', 'g'),
    ('uav3', 'r'),
    ('uav4', 'c'),
    ('uav5', 'm'),
    ('uav6', 'y'),
    ('uav7', 'k'),
    ('uav8', 'orange'),
    ('uav9', 'purple'),
    ('uav10', 'brown')
    ])
    
'''
uav_positions = OrderedDict((uav, []) for uav in uav_colors)

# Initialize UAV trajectories using OrderedDict
uav_trajectories = OrderedDict((uav, {'x': [], 'y': [], 'color': color}) for uav, color in uav_colors.items())
'''
print("uavs_array",uavs_array)
uav_positions = OrderedDict((uav, []) for uav in uavs_array	)

print("uav_positions",uav_positions)
uav_trajectories = OrderedDict()
for i, uav in enumerate(uavs_array):
    if uav in uav_colors:
        color = uav_colors[uav]
    else:
        # If there's no predefined color for the UAV, you might want to assign a default color
        color = 'gray'  # Example: assigning a default color

    uav_trajectories[uav] = {'x': [], 'y': [], 'color': color}
    
for i, (uav, trajectory) in enumerate(uav_trajectories.items()):
    if i < len(uavs_array):
	uav_trajectories[uav]['color'] = uav_colors[uavs_array[i]]
print("uav_trajectories",uav_trajectories)

def update_plot():	
	global area_covered_var,fig,ax,fileloc,filename,filepath,uav_trajectories,uav_colors,uavs_array, home_pos,goal_points,uav_positions
	uav_positions = OrderedDict((uav, []) for uav in uavs_array	)

	print("uav_positions",uav_positions)
	uav_trajectories = OrderedDict()
	for i, uav in enumerate(uavs_array):
	    if uav in uav_colors:
		color = uav_colors[uav]
	    else:
		# If there's no predefined color for the UAV, you might want to assign a default color
		color = 'gray'  # Example: assigning a default color

	    uav_trajectories[uav] = {'x': [], 'y': [], 'color': color}
	for i, (uav, trajectory) in enumerate(uav_trajectories.items()):
	    if i < len(uavs_array):
		uav_trajectories[uav]['color'] = uav_colors[uavs_array[i]]
	print("uav_trajectories",uav_trajectories)

	print("*****.....update_plot")
	#plt.close('all')
 
	#uav_positions = {'uav_1': [], 'uav_2': [], 'uav_3': [], 'uav_4': [], 'uav_5': [], 'uav_6':[], 'uav_7':[], 'uav_8':[], 'uav_9':[], 'uav_10':[]}
	start_time = time.time()
	elapsed_time = 0
	file_sock2.sendto(str(filename).encode(),file_server_address2)
	time.sleep(0.5)
	file_sock3.sendto(str(filename).encode(),file_server_address3)
	time.sleep(0.5)
	file_sock4.sendto(str(filename).encode(),file_server_address4)
	time.sleep(0.5)
	file_sock9.sendto(str(filename).encode(),file_server_address9)
	'''
	time.sleep(0.5)
	file_sock4.sendto(str(filename).encode(),file_server_address4)
	time.sleep(0.5)
	file_sock5.sendto(str(filename).encode(),file_server_address5)
	time.sleep(0.5)
	file_sock6.sendto(str(filename).encode(),file_server_address6)
	time.sleep(0.5)
	file_sock7.sendto(str(filename).encode(),file_server_address7)
	time.sleep(0.5)
	file_sock8.sendto(str(filename).encode(),file_server_address8)
	time.sleep(0.5)
	file_sock9.sendto(str(filename).encode(),file_server_address9)
	'''
	size,obs=update_load_yaml(filepath)
	#ax.clear()
	print("%%%%%%%%%%%%")
	canvas = FigureCanvasTkAgg(fig, master=root)
	canvas_widget = canvas.get_tk_widget()
	#canvas.draw()
	canvas_widget.place(x=1000, y=30, width=900, height=800)
	#canvas.get_tk_widget().place(x=1000, y=30, width=900, height=870)
	toolbar = NavigationToolbar2Tk(canvas, root)
	toolbar.update()
	# Place the toolbar in your Tkinter window
	toolbar.place(x=1300, y=870, width=900, height=50)		
	def update_plot1(canvas):
		count=0
		global area_covered_var,fig,ax,fileloc,filename,filepath,uav_trajectories,uav_colors,uav_positions,follower_host_tuple,goal_table,return_goal_table
		while True:
			
			#print("filepath",filepath)
			try:
				data, address = sock.recvfrom(1024)
				if (data==b"start") or (data==b"aggregate") or (data==b"return") or (data==b"same altitude") or (data==b"different altitude") or (data==b"disperse") or (data==b"stop")  or(data==b"search") or (data==b"circle formation") or (data.startswith(b"Drone"))or (data.startswith(b"vehicle"))  or (data.startswith(b"Vehicle")) or (data.startswith(b"master_num"))or (data.startswith(b"pos_array")) or (data.startswith(b"home")) or (data=="rtl") or (data==b"goal") or (data==b"specific_bot_goal") or (data==b"CSV Cleared") or (data==b"Strike Cancelled") or (data.startswith(b"remove_bot")) or (data.endswith(b"vehicle removed")):
					print(str(data)," command received")
					msg=str(data)
					terminal_message_entry.insert(tk.END,msg+'\n')
					terminal_message_entry.see(tk.END)
					root.update()
					#continue
				if(data.startswith(b"home_pos")):
					message_with_array = data.decode()
					message = message_with_array[:8]  # Assuming "home_pos" is 8 characters long
					array_data = message_with_array[8:]	

					# Deserialize the array data
					home_pos = json.loads(array_data)
					terminal_message_entry.insert(tk.END,str(data)+'\n')
					terminal_message_entry.see(tk.END)
					root.update()

				# Print the received array and message
					print("Received message:",message)
					print("Received array:home_pos", home_pos)
					print("home_pos!",home_pos[0][0])
			     	
					#continue
			     
				if(data.startswith(b"goal_points")):
					message_with_array = data.decode()
					message = message_with_array[:11] # Assuming "home_pos" is 8 characters long
					array_data = message_with_array[11:]	

				# Deserialize the array data
					goal_points = json.loads(array_data)
					terminal_message_entry.insert(tk.END,str(data)+'\n')
					terminal_message_entry.see(tk.END)
					root.update()

				# Print the received array and message
					print("Received message:", message)
					print("Received array:GOALPOINTS##################", goal_points)
					print("goal_points!",goal_points[0][0],goal_points[0][1])
			     	
					#continue
			     
				if data.startswith(b"search,"):
					
					a = data.decode('utf8').split(',')
					area_covered = (a[1])
					#area_covered=float(area_covered)*10
					print("area_covered",area_covered)
					search_time = (a[2])
					print("search_time",search_time)
			     	#area_percentage = (area_covered / total_area) * 100
					minutes=int(float(search_time) // 60)
					seconds=int(float(search_time) % 60)
					timer_text = "Time: {int(float(search_time) // 60):02d}:{int(float(search_time) % 60):02d}"
					print(minutes,":",seconds, "area_covered:", area_covered,search_time)
					#label.config(text="{timer_text}   area_covered: {area_covered}")
					area_covered_var.set("Area Covered : %s       Time: %02d:%02d" % (area_covered, minutes, seconds))
					
				#area_covered_var.set("Area Covered = " + str(area_covered)+ "Time"+{minutes:02d}+":"+{seconds:02d})
					#continue
				'''
				a = data.decode('utf8').split(',')
				a = [value.strip('{}') for value in a]  # Remove curly braces
				#print("a",a)
				
				for uav in uav_positions:
					print("uav",uav)
					uav_positions.append((float(a.pop(0)), float(a.pop(0))))
					print("!!!!uav_positions!!",uav_positions)
				'''
				a = data.decode('utf8').split(',')
				#print('a',a[-1],type(a[-1]))
				
				if a[-1].strip().startswith(b'path'):
				# Remove the last element (which is "path4")
					print("**********")
					last_element = a.pop()
					print("last_element",last_element,type(last_element))
					if last_element.strip().startswith('path'):
						print("Removed element:", last_element[4:])
						h=int(last_element[4:])
						print("HHHHHHHHHHHH",h)
						'''
						if h in goal_table:
						    # Remove the value from the table
						    goal_table = [v for v in goal_table if v != h]
						#print("goal_path_csv_array",goal_path_csv_array)
						'''
					
						if h not in goal_table:
							print("h$$$$$$$$$$$", h)
							goal_table.append(h)
						
						#goal_table.append(int(last_element[4:]))
						print('goal_table!!!!!',goal_table)
					print("a without path",a)
					print('goal_table!!!!!',goal_table)
				if a[-1].strip().startswith(b'return_path'):
				# Remove the last element (which is "path4")
					print("**********")
				        last_element = a.pop()
					print("last_element",last_element,type(last_element))
					if last_element.strip().startswith(b'return_path'):
						print("PERIIIIIIIIIIIIYAAAAAAAAAAAAA pRIntttttttttt")
						print("Removed element:", last_element[11:])
						h=int(last_element[11:])
						print("HHHHHHHHHHHH",h)
						'''
						if h in goal_table:
						    # Remove the value from the table
						    goal_table = [v for v in goal_table if v != h]
						#print("goal_path_csv_array",goal_path_csv_array)
						'''
						if h not in return_goal_table:
							print("h$$$$$$$$$$$", h)
							return_goal_table.append(h)
						print("return_goal_table",return_goal_table)
					print("return_goal_table",return_goal_table)
				print("return_goal_table>>>>>>>>>!!!!!!!!!!!!!!!",goal_table,return_goal_table)
				a = [value.strip('{}') for value in a]  # Remove curly braces

				for uav, positions in uav_positions.items():
					positions.append((float(a.pop(0)), float(a.pop(0))))
                # Update UAV trajectories
					uav_trajectories[uav]['x'].append(positions[-1][0])
					uav_trajectories[uav]['y'].append(positions[-1][1])

				ax.clear()
				ax.relim()
				ax.autoscale_view()
				#print("home_pos_flag",home_pos_flag)
				if home_pos_flag:
					marked_points_x = [point[0] for point in home_pos]
					marked_points_y = [point[1] for point in home_pos]
				    	for i, (x, y) in enumerate(home_pos):
						ax.scatter(x, y, marker="${}$".format(uavs_array[i]), color="red", s=300)
				#print("goal_points_flag",goal_points_flag)
				
				if goal_points_flag:
					marked_points_x = [point[0] for point in goal_points]
					marked_points_y = [point[1] for point in goal_points]
				    	for i, (x, y) in enumerate(goal_points, start=1):
						#ax.scatter(x, y, marker="${}$".format(i), color="red", s=100, label="Goal Pos {}".format(i))
					 	ax.scatter(x, y, marker="${}$".format(i), color="red", s=100)
					for i, (x, y) in enumerate(home_pos):
					 	ax.scatter(x, y, marker="${}$".format("H{}".format(i+1)), color="blue", s=100)
				
				#print("uav_trajectories",uav_trajectories)
				
				for uav, trajectory in uav_trajectories.items():
					x, y, color = trajectory['x'], trajectory['y'], trajectory['color']
					ax.plot(x, y, linestyle='-', color=color, alpha=0.5)
					ax.plot(x[-1], y[-1], marker='o', label=uav, color=color)
			
				for obs in obstacles:
					x,y = obs.exterior.xy
					ax.fill(x,y, fc='gray', alpha=0.9)

				ax.set_xlim(0, size[0])
				ax.set_ylim(0, size[1])
				#ax.pause(0.05)
				ax.relim()
				ax.legend()
				ax.autoscale_view()
				fig.canvas.draw()
				fig.canvas.flush_events()
				canvas.draw_idle()		     	
			except:
			 	pass
			root.update()	
	update_plot1(canvas)


def goal_points_func():
        print("%^^^^^^^^^^%")
	global goal_points_flag,goal_points,fig,ax,goal_checkbox_var
	goal_points_flag=goal_checkbox_var.get() == 1
	print("goal_points_flag",goal_points_flag)


	
def toggle_home_point():
    global home_pos,home_pos_flag,fig,ax,home_checkbox_var
    home_pos_flag=home_checkbox_var.get() == 1
    #home_pos_flag=True
    print("home_pos_flag",home_pos_flag)

def clear_home_pos_markers():
    ax.clear()
    update_plot()
    
def clear_csv():
	print("!!!CSV Cleared!!")
        global udp_socket,udp_socket2,server_address1,server_address2
        #udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = "clear_csv"
        '''
        udp_socket.sendto(str(data).encode(), server_address1)
        time.sleep(0.5)
        udp_socket2.sendto(str(data).encode(), server_address2)
        time.sleep(0.5)
	'''
        udp_socket3.sendto(str(data).encode(), server_address3)
        time.sleep(0.5)
        udp_socket4.sendto(str(data).encode(), server_address4)
        time.sleep(0.5)
        udp_socket5.sendto(str(data).encode(), server_address5)
        time.sleep(0.5)
        '''
        udp_socket6.sendto(str(data).encode(), server_address6)
        time.sleep(0.5)
        udp_socket7.sendto(str(data).encode(), server_address7)
        time.sleep(0.5)
        udp_socket8.sendto(str(data).encode(), server_address8)
        time.sleep(0.5)
        udp_socket9.sendto(str(data).encode(), server_address9)
        time.sleep(0.5)
        '''
        udp_socket10.sendto(str(data).encode(), server_address10)
        
        
def strike_target():
	    # Retrieve values from entry widgets
	    model_no = selected_drone.get()
	    altitude = c_alt_entry.get()
	    target_value = target_entry.get()
	    rtl_height = rtl_height_entry.get()

	    # Perform action, e.g., strike the target
	    print("Striking target:")
	    print("Model No:", model_no)
	    print("Altitude:", altitude)
	    print("Target Value:", target_value,type(target_value))
	    print("RTL Height:", rtl_height)
	    
	    val=str(model_no)+","+str(altitude)+","+str(target_value)+","+str(rtl_height)
	    print("val",val)
	    d="strike"+","+val
	    print("d",d)
	    '''
	    udp_socket.sendto(str(d).encode(), server_address1)
	    time.sleep(0.5)
	    udp_socket2.sendto(str(d).encode(), server_address2)
	    time.sleep(0.5)
	    '''
	    udp_socket3.sendto(str(d).encode(), server_address3)
	    time.sleep(0.5)
	    udp_socket4.sendto(str(d).encode(), server_address4)
	    time.sleep(0.5)
	    udp_socket5.sendto(str(d).encode(), server_address5)
	    time.sleep(0.5)
	    '''
	    udp_socket6.sendto(str(d).encode(), server_address6)
	    time.sleep(0.5)
	    udp_socket7.sendto(str(d).encode(), server_address7)
	    time.sleep(0.5)
	    udp_socket8.sendto(str(d).encode(), server_address8)
	    time.sleep(0.5)
	    udp_socket9.sendto(str(d).encode(), server_address9)
	    time.sleep(0.5)
	    '''
	    udp_socket10.sendto(str(d).encode(), server_address10)
	    
def strike_stop():
	print("Strike STOP!!!!!!!")
        global socket1,udp_socket,udp_socket2,server_address1,server_address2      
        socket1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = "strike_stop"
        print("data",type(data))
        '''
        socket1.sendto(str(data).encode(), server_address11)
        time.sleep(0.5)
        socket2.sendto(str(data).encode(), server_address12)
        time.sleep(0.5)
        '''
        socket3.sendto(str(data).encode(), server_address13)
        time.sleep(0.5)
        socket4.sendto(str(data).encode(), server_address14)
        time.sleep(0.5)
        socket5.sendto(str(data).encode(), server_address15)
        time.sleep(0.5)
        '''
        socket6.sendto(str(data).encode(), server_address16)
        time.sleep(0.5)
       
        socket7.sendto(str(data).encode(), server_address17)
        time.sleep(0.5)
        socket8.sendto(str(data).encode(), server_address18)
        time.sleep(0.5)
        socket9.sendto(str(data).encode(), server_address19)
        time.sleep(0.5)
        '''
        socket10.sendto(str(data).encode(), server_address20)
    
def start_socket():
        print("!!!Start!!")
        global udp_socket,udp_socket2,server_address1,server_address2
        #udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = "start"
        '''
        udp_socket.sendto(str(data).encode(), server_address1)
        time.sleep(0.5)
        udp_socket2.sendto(str(data).encode(), server_address2)
        time.sleep(0.5)
	'''
        udp_socket3.sendto(str(data).encode(), server_address3)
        time.sleep(0.5)
        udp_socket4.sendto(str(data).encode(), server_address4)
        time.sleep(0.5)
        udp_socket5.sendto(str(data).encode(), server_address5)
        time.sleep(0.5)
        '''
        udp_socket6.sendto(str(data).encode(), server_address6)
        time.sleep(0.5)
        udp_socket7.sendto(str(data).encode(), server_address7)
        time.sleep(0.5)
        udp_socket8.sendto(str(data).encode(), server_address8)
        time.sleep(0.5)
        udp_socket9.sendto(str(data).encode(), server_address9)
        time.sleep(0.5)
        '''
        udp_socket10.sendto(str(data).encode(), server_address10)
        
        
def start1_socket():
        print("Start1........")
        global udp_socket,server_address1,server_address2,udp_socket2
        #udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = "start1"
        '''
        udp_socket.sendto(str(data).encode(), server_address1)
        time.sleep(0.5)
        udp_socket2.sendto(str(data).encode(), server_address2)
        time.sleep(0.5)
        '''
        udp_socket3.sendto(str(data).encode(), server_address3)
        time.sleep(0.5)
        udp_socket4.sendto(str(data).encode(), server_address4)
        time.sleep(0.5)
        udp_socket5.sendto(str(data).encode(), server_address5)
        time.sleep(0.5)
        '''
        udp_socket6.sendto(str(data).encode(), server_address6)
        time.sleep(0.5)
        udp_socket7.sendto(str(data).encode(), server_address7)
        time.sleep(0.5)
        udp_socket8.sendto(str(data).encode(), server_address8)
        time.sleep(0.5)
        udp_socket9.sendto(str(data).encode(), server_address9)
        time.sleep(0.5)
        '''
        udp_socket10.sendto(str(data).encode(), server_address10)
     
def home_lock():
        print("Home position Locked....!!!!")
        global udp_socket,server_address1,server_address2,udp_socket2
        #udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = "home_lock"
        '''
        udp_socket.sendto(str(data).encode(), server_address1)
        time.sleep(0.5)
        udp_socket2.sendto(str(data).encode(), server_address2)
        time.sleep(0.5)
        '''
        udp_socket3.sendto(str(data).encode(), server_address3)
        time.sleep(0.5)
        udp_socket4.sendto(str(data).encode(), server_address4)
        time.sleep(0.5)
        udp_socket5.sendto(str(data).encode(), server_address5)
        time.sleep(0.5)
        '''
        udp_socket6.sendto(str(data).encode(), server_address6)
        time.sleep(0.5)
        udp_socket7.sendto(str(data).encode(), server_address7)
        time.sleep(0.5)
        udp_socket8.sendto(str(data).encode(), server_address8)
        time.sleep(0.5)
        udp_socket9.sendto(str(data).encode(), server_address9)
        time.sleep(0.5)
        '''
        udp_socket10.sendto(str(data).encode(), server_address10)
        
        
def share_data_func():
	global goal_table,return_goal_table,filename,udp_socket,server_address1,server_address2,udp_socket2
	print("*******",goal_table)
	#goal_table=[]
	
	if return_goal_table is not None:
		combined_goal_table = goal_table + return_goal_table
		print("combined_goal_table",combined_goal_table)
		'''
		udp_socket.sendto(("share_data" + "," + str(combined_goal_table)).encode(), server_address1)
		time.sleep(0.5)
		udp_socket2.sendto(("share_data" + "," + str(combined_goal_table)).encode(), server_address2)
		time.sleep(0.5)
		'''
		udp_socket3.sendto(("share_data" + "," + str(combined_goal_table)).encode(), server_address3)
		time.sleep(0.5)
		udp_socket4.sendto(("share_data" + "," + str(combined_goal_table)).encode(), server_address4)
		time.sleep(0.5)
		udp_socket5.sendto(("share_data" + "," + str(combined_goal_table)).encode(), server_address5)
		time.sleep(0.5)
		'''
		udp_socket6.sendto(("share_data" + "," + str(combined_goal_table)).encode(), server_address6)
		time.sleep(0.5)
		udp_socket7.sendto(("share_data" + "," + str(combined_goal_table)).encode(), server_address7)
		time.sleep(0.5)
		udp_socket8.sendto(("share_data" + "," + str(combined_goal_table)).encode(), server_address8)
		time.sleep(0.5)
		udp_socket9.sendto(("share_data" + "," + str(combined_goal_table)).encode(), server_address9)
		time.sleep(0.5)
		'''
		udp_socket10.sendto(("share_data" + "," + str(combined_goal_table)).encode(), server_address10)
        else:
		'''
		udp_socket.sendto(("share_data"+","+str(goal_table)).encode(), server_address1)
		time.sleep(0.5)
		udp_socket2.sendto(("share_data"+","+str(goal_table)).encode(), server_address2)
		time.sleep(0.5)
		'''
		udp_socket3.sendto(("share_data"+","+str(goal_table)).encode(), server_address3)
		time.sleep(0.5)
		udp_socket4.sendto(("share_data"+","+str(goal_table)).encode(), server_address4)
		time.sleep(0.5)
		udp_socket5.sendto(("share_data"+","+str(goal_table)).encode(), server_address5)
		time.sleep(0.5)
		'''
		udp_socket6.sendto(("share_data"+","+str(goal_table)).encode(), server_address6)
		time.sleep(0.5)
		udp_socket7.sendto(("share_data"+","+str(goal_table)).encode(), server_address7)
		time.sleep(0.5)
		udp_socket8.sendto(("share_data"+","+str(goal_table)).encode(), server_address8)
		time.sleep(0.5)
		udp_socket9.sendto(("share_data"+","+str(goal_table)).encode(), server_address9)
		time.sleep(0.5)
		'''
		udp_socket10.sendto(("share_data"+","+str(goal_table)).encode(), server_address10)
	
        
def disperse_socket(): 
        global udp_socket,server_address1,server_address2,udp_socket2
        print("Disperse!!!!!!")
        #udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = "disperse"
        '''
        udp_socket.sendto(str(data).encode(), server_address1)
        time.sleep(0.5)
        udp_socket2.sendto(str(data).encode(), server_address2)
        time.sleep(0.5)
        '''
        udp_socket3.sendto(str(data).encode(), server_address3)
        time.sleep(0.5)
        udp_socket4.sendto(str(data).encode(), server_address4)
        time.sleep(0.5)
        udp_socket5.sendto(str(data).encode(), server_address5)
        time.sleep(0.5)
        '''
        udp_socket6.sendto(str(data).encode(), server_address6)
        time.sleep(0.5)
        udp_socket7.sendto(str(data).encode(), server_address7)
        time.sleep(0.5)
        udp_socket8.sendto(str(data).encode(), server_address8)
        time.sleep(0.5)
        udp_socket9.sendto(str(data).encode(), server_address9)
        time.sleep(0.5)
        '''
        udp_socket10.sendto(str(data).encode(), server_address10)
        
        
def takeoff_socket():
	print("Takeoff...........")
	takeoff_alt = takeoff_entry.get()
	try:
		takeoff_alt = int(takeoff_alt)
	except:
		takeoff_alt = int(10)
        global udp_socket,server_address1,server_address2,udp_socket2
        #udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = "takeoff"+","+str(takeoff_alt)
        '''
        sent = udp_socket.sendto(str(data).encode(), server_address1)
        time.sleep(0.5)
        sent = udp_socket2.sendto(str(data).encode(), server_address2)
        time.sleep(0.5)
        '''
        sent = udp_socket3.sendto(str(data).encode(), server_address3)
        time.sleep(0.5)
        sent = udp_socket4.sendto(str(data).encode(), server_address4)
        time.sleep(0.5)
        sent = udp_socket5.sendto(str(data).encode(), server_address5)
        '''
        time.sleep(0.5)
        sent = udp_socket6.sendto(str(data).encode(), server_address6)
        time.sleep(0.5)
        sent = udp_socket7.sendto(str(data).encode(), server_address7)
        time.sleep(0.5)
        sent = udp_socket8.sendto(str(data).encode(), server_address8)
        time.sleep(0.5)
        sent = udp_socket9.sendto(str(data).encode(), server_address9)
        '''
        time.sleep(0.5)
        sent = udp_socket10.sendto(str(data).encode(), server_address10)
        
def search_socket(): 
        global udp_socket,server_address1,server_address2,udp_socket2
        print("Searching........")
        #udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = "search"
        '''
        udp_socket.sendto(str(data).encode(), server_address1)
        time.sleep(0.5)
        udp_socket2.sendto(str(data).encode(), server_address2)
        time.sleep(0.5)
        '''
        udp_socket3.sendto(str(data).encode(), server_address3)
        time.sleep(0.5)
        udp_socket4.sendto(str(data).encode(), server_address4)
        time.sleep(0.5)
        udp_socket5.sendto(str(data).encode(), server_address5)
        '''
        time.sleep(0.5)
        udp_socket6.sendto(str(data).encode(), server_address6)
        time.sleep(0.5)
        udp_socket7.sendto(str(data).encode(), server_address7)
        time.sleep(0.5)
        udp_socket8.sendto(str(data).encode(), server_address8)
        time.sleep(0.5)
        udp_socket9.sendto(str(data).encode(), server_address9)
        '''
        time.sleep(0.5)
        udp_socket10.sendto(str(data).encode(), server_address10)
        
def aggregate_socket():
        global udp_socket,server_address1,server_address2,udp_socket2
        print("Aggregation..!!!!")
        #udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = "aggregate"
        '''
        udp_socket.sendto(str(data).encode(), server_address1)
        time.sleep(0.5)
        udp_socket2.sendto(str(data).encode(), server_address2)
        time.sleep(0.5)
        '''
        udp_socket3.sendto(str(data).encode(), server_address3)
        time.sleep(0.5)
        udp_socket4.sendto(str(data).encode(), server_address4)
        time.sleep(0.5)
        udp_socket5.sendto(str(data).encode(), server_address5)
        '''
        time.sleep(0.5)      
        udp_socket6.sendto(str(data).encode(), server_address6)
        time.sleep(0.5)
        udp_socket7.sendto(str(data).encode(), server_address7)
        time.sleep(0.5)
        udp_socket8.sendto(str(data).encode(), server_address8)
        time.sleep(0.5)
        udp_socket9.sendto(str(data).encode(), server_address9)
        '''
        time.sleep(0.5)
        udp_socket10.sendto(str(data).encode(), server_address10)
        
def home_socket(): 
	print("Home....******")
        global udp_socket,server_address1,server_address2,udp_socket2
        #udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = "home"
        '''
        udp_socket.sendto(str(data).encode(), server_address1)
        time.sleep(0.5)
        udp_socket2.sendto(str(data).encode(), server_address2)
        time.sleep(0.5)
        udp_socket3.sendto(str(data).encode(), server_address3)
        time.sleep(0.5)
        '''
        udp_socket4.sendto(str(data).encode(), server_address4)
        time.sleep(0.5)
        udp_socket5.sendto(str(data).encode(), server_address5)
        time.sleep(0.5)
        udp_socket6.sendto(str(data).encode(), server_address6)
        time.sleep(0.5)
        '''
        udp_socket7.sendto(str(data).encode(), server_address7)
        time.sleep(0.5)
        udp_socket8.sendto(str(data).encode(), server_address8)
        time.sleep(0.5)
        udp_socket9.sendto(str(data).encode(), server_address9)
        '''
        time.sleep(0.5)
        udp_socket10.sendto(str(data).encode(), server_address10)
        
        
def different_alt_socket():
	global  udp_socket,server_address1,server_address2,udp_socket2
	print("Hello")
        #udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data =altd_entry.get()
        print(data)
        g="different"+","+str(data)
        print(g,"data")
        '''
        udp_socket.sendto(str(g).encode(), server_address1)
        time.sleep(0.5)
        udp_socket2.sendto(str(g).encode(), server_address2)
        time.sleep(0.5)
        '''
        udp_socket3.sendto(str(g).encode(), server_address3)
        time.sleep(0.5)
        udp_socket4.sendto(str(g).encode(), server_address4)
        time.sleep(0.5)
        udp_socket5.sendto(str(g).encode(), server_address5)
        time.sleep(0.5)
        '''
        udp_socket6.sendto(str(g).encode(), server_address6)
        time.sleep(0.5)
        udp_socket7.sendto(str(g).encode(), server_address7)
        time.sleep(0.5)
        udp_socket8.sendto(str(g).encode(), server_address8)
        time.sleep(0.5)
        udp_socket9.sendto(str(g).encode(), server_address9)
        '''
        time.sleep(0.5)
        udp_socket10.sendto(str(g).encode(), server_address10)
        
def same_alt_socket():
	global udp_socket,server_address1,server_address2,udp_socket2
	print("Same_altitude")
       # udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = alts_entry.get()
        print(data,"data")
        f="same"+","+str(data)
        print(f,"f")
        '''
        udp_socket.sendto(str(f).encode(), server_address1)
        time.sleep(0.5)
        udp_socket2.sendto(str(f).encode(), server_address2)
        time.sleep(0.5)
        '''
        udp_socket3.sendto(str(f).encode(), server_address3)
        time.sleep(0.5)
        udp_socket4.sendto(str(f).encode(), server_address4)
        time.sleep(0.5)
        udp_socket5.sendto(str(f).encode(), server_address5)
        '''
        time.sleep(0.5)
        udp_socket6.sendto(str(f).encode(), server_address6)
        time.sleep(0.5)
        udp_socket7.sendto(str(f).encode(), server_address7)
        time.sleep(0.5)
        udp_socket8.sendto(str(f).encode(), server_address8)
        time.sleep(0.5)
        udp_socket9.sendto(str(f).encode(), server_address9)
        '''
        time.sleep(0.5)
        udp_socket10.sendto(str(f).encode(), server_address10)
        #socket.close()
   
def home_goto_socket(): 
        global socket,udp_socket,server_address1,server_address2,udp_socket2
        #udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = "home_goto"
        '''
        udp_socket.sendto(str(data).encode(), server_address1)
        time.sleep(0.5)
        udp_socket2.sendto(str(data).encode(), server_address2)
        time.sleep(0.5)
        '''
        udp_socket3.sendto(str(data).encode(), server_address3)
        time.sleep(0.5)
        udp_socket4.sendto(str(data).encode(), server_address4)
        time.sleep(0.5)
        udp_socket5.sendto(str(data).encode(), server_address5)
        '''
        time.sleep(0.5)
        udp_socket6.sendto(str(data).encode(), server_address6)
        time.sleep(0.5)
        udp_socket7.sendto(str(data).encode(), server_address7)
        time.sleep(0.5)
        udp_socket8.sendto(str(data).encode(), server_address8)
        time.sleep(0.5)
        udp_socket9.sendto(str(data).encode(), server_address9)
        '''
        time.sleep(0.5)
        udp_socket10.sendto(str(data).encode(), server_address10)
        
def rtl_socket():
        global socket,udp_socket,server_address1,server_address2,udp_socket2
        #udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = "rtl"
        '''
        udp_socket.sendto(str(data).encode(), server_address1)
        time.sleep(0.5)
        udp_socket2.sendto(str(data).encode(), server_address2)
        time.sleep(0.5)
        '''
        udp_socket3.sendto(str(data).encode(), server_address3)
        time.sleep(0.5)
        udp_socket4.sendto(str(data).encode(), server_address4)
        time.sleep(0.5)
        udp_socket5.sendto(str(data).encode(), server_address5)
        time.sleep(0.5)
        '''
        udp_socket6.sendto(str(data).encode(), server_address6)
        time.sleep(0.5)
        udp_socket7.sendto(str(data).encode(), server_address7)
        time.sleep(0.5)
        udp_socket8.sendto(str(data).encode(), server_address8)
        time.sleep(0.5)
        udp_socket9.sendto(str(data).encode(), server_address9)
        '''
        time.sleep(0.5)
        udp_socket10.sendto(str(data).encode(), server_address10)
        
def stop_socket():
	print("STOP>>>>>>>>>>>>")
        global socket1,udp_socket,udp_socket2,server_address1,server_address2      
        socket1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = "stop"
        '''
        socket1.sendto(str(data).encode(), server_address11)
        time.sleep(0.5)
        socket2.sendto(str(data).encode(), server_address12)
        time.sleep(0.5)
        '''
        socket3.sendto(str(data).encode(), server_address13)
        time.sleep(0.5)
        socket4.sendto(str(data).encode(), server_address14)
        time.sleep(0.5)
        socket5.sendto(str(data).encode(), server_address15)
        '''
        time.sleep(0.5)
        socket6.sendto(str(data).encode(), server_address16)
        time.sleep(0.5)
        socket7.sendto(str(data).encode(), server_address17)
        time.sleep(0.5)
        socket8.sendto(str(data).encode(), server_address18)
        time.sleep(0.5)
        socket9.sendto(str(data).encode(), server_address19)
        '''
        time.sleep(0.5)
        socket10.sendto(str(data).encode(), server_address20)
        
def move_bot_stop():
	print("Move STOP>>>>>>>>>>>>")
        global socket1,udp_socket,udp_socket2,server_address1,server_address2      
        socket1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = "move_bot_stop"
        '''
        socket1.sendto(str(data).encode(), server_address11)
        time.sleep(0.5)
        socket2.sendto(str(data).encode(), server_address12)
        time.sleep(0.5)
        '''
        socket3.sendto(str(data).encode(), server_address13)
        time.sleep(0.5)
        socket4.sendto(str(data).encode(), server_address14)
        time.sleep(0.5)
        socket5.sendto(str(data).encode(), server_address15)
        '''
        time.sleep(0.5)
        socket6.sendto(str(data).encode(), server_address16)
        time.sleep(0.5)
        socket7.sendto(str(data).encode(), server_address17)
        time.sleep(0.5)
        socket8.sendto(str(data).encode(), server_address18)
        time.sleep(0.5)
        socket9.sendto(str(data).encode(), server_address19)
        '''
        time.sleep(0.5)
        socket10.sendto(str(data).encode(), server_address20)

def land_socket(): 
	global follower_host_tuple,socket,udp_socket,server_address1,server_address2,udp_socket2
        data = "land"
        '''
        udp_socket.sendto(str(data).encode(), server_address1)
        time.sleep(0.5)
        udp_socket2.sendto(str(data).encode(), server_address2)
        time.sleep(0.5)
        '''
        udp_socket3.sendto(str(data).encode(), server_address3)
        time.sleep(0.5)
        udp_socket4.sendto(str(data).encode(), server_address4)
        time.sleep(0.5)
        udp_socket5.sendto(str(data).encode(), server_address5)
        '''
        time.sleep(0.5)
        udp_socket6.sendto(str(data).encode(), server_address6)
        time.sleep(0.5)
        udp_socket7.sendto(str(data).encode(), server_address7)
        time.sleep(0.5)
        udp_socket8.sendto(str(data).encode(), server_address8)
        time.sleep(0.5)
        udp_socket9.sendto(str(data).encode(), server_address9)
        '''
        time.sleep(0.5)
        udp_socket10.sendto(str(data).encode(), server_address10)
	for i,e in enumerate(follower_host_tuple):
		print("iiiiiiii",i)
		follower_host_tuple[i].mode=VehicleMode("LAND")
		#follower_host_tuple[i].close()		
	
def return_socket():
        global socket,return_flag,udp_socket,server_address1,server_address2,udp_socket2
        #print("&&&&&")
        return_flag=True
        #udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = "return"
        '''
        udp_socket.sendto(str(data).encode(), server_address1)
        time.sleep(0.5)
        udp_socket2.sendto(str(data).encode(), server_address2)
        time.sleep(0.5)
        '''
        udp_socket3.sendto(str(data).encode(), server_address3)
        time.sleep(0.5)
        udp_socket4.sendto(str(data).encode(), server_address4)
        time.sleep(0.5)
        udp_socket5.sendto(str(data).encode(), server_address5)
        time.sleep(0.5)
        '''
        udp_socket6.sendto(str(data).encode(), server_address6)
        time.sleep(0.5)
        udp_socket7.sendto(str(data).encode(), server_address7)
        time.sleep(0.5)
        udp_socket8.sendto(str(data).encode(), server_address8)
        time.sleep(0.5)
        udp_socket9.sendto(str(data).encode(), server_address9)
        time.sleep(0.5)
        '''
        udp_socket10.sendto(str(data).encode(), server_address10)
'''        
def resume_socket():
        global socket1
	#print("&&&&&")
	socket1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	data = "resume"
	socket1.sendto(str(data).encode(), server_address11)
	#socket1.sendto(str(data).encode(), server_address12)
	#socket1.sendto(str(data).encode(), server_address13)
'''	
def specific_bot_goal_socket(): 
	print("$$$##Specific_bot_goal###")
        global socket,udp_socket,server_address1,server_address2,udp_socket2
        #udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = specific_bot_goal_entry.get()
        print(data,type(data),"KKKKK")
        d="specific_bot_goal"+","+str(data)
        print("d",d)
        '''
        udp_socket.sendto(str(d).encode(), server_address1)
        time.sleep(0.5)
        udp_socket2.sendto(str(d).encode(), server_address2)
        time.sleep(0.5)
        '''
        udp_socket3.sendto(str(d).encode(), server_address3)
        time.sleep(0.5)
        udp_socket4.sendto(str(d).encode(), server_address4)
        time.sleep(0.5)
        udp_socket5.sendto(str(d).encode(), server_address5)
        time.sleep(0.5)
        '''
        udp_socket6.sendto(str(d).encode(), server_address6)
        time.sleep(0.5)
        udp_socket7.sendto(str(d).encode(), server_address7)
        time.sleep(0.5)
        udp_socket8.sendto(str(d).encode(), server_address8)
        time.sleep(0.5)
        udp_socket9.sendto(str(d).encode(), server_address9)
        time.sleep(0.5)
        '''
        udp_socket10.sendto(str(d).encode(), server_address10)
        specific_bot_goal_entry.bind("<Return>",specific_bot_goal_socket)
       
def fetch_and_compute():
    # Fetch the data from uavs_entry
    uavs_value = uavs_entry.get()
    
    # Convert the fetched value to an integer
    uavs_value = int(uavs_value)
    
    # Check if the entered value is 3
    if uavs_value == 3:
        # Compute the result
        computed_result = 12
        # Display the computed value in computed_entry
        computed_entry.delete(0, tk.END)  # Clear the entry box
        computed_entry.insert(0, computed_result)
    elif uavs_value == 5:
        # Compute the result
        computed_result = 10
        # Display the computed value in computed_entry
        computed_entry.delete(0, tk.END)  # Clear the entry box
        computed_entry.insert(0, computed_result)
    elif uavs_value == 7:
        # Compute the result
        computed_result = 8
        # Display the computed value in computed_entry
        computed_entry.delete(0, tk.END)  # Clear the entry box
        computed_entry.insert(0, computed_result)
        
    elif uavs_value == 10:
        # Compute the result
        computed_result = 6
        # Display the computed value in computed_entry
        computed_entry.delete(0, tk.END)  # Clear the entry box
        computed_entry.insert(0, computed_result)
     
'''
def fetch_and_compute():
    # Fetch the data from uavs_entry
    uavs_value = uavs_entry.get()
    
    # Check if the entry is empty
    if not uavs_value:
        computed_entry.delete(0, tk.END)
        computed_entry.insert(0, "Please enter a value")
        return
    
    try:
        # Convert the fetched value to an integer
        uavs_value = int(uavs_value)
        
        # Compute the time based on the number of UAVs
        # Example formula: time taken = 24 / (number of UAVs)
        computed_result = 24 / uavs_value
        
        # Display the computed value in computed_entry
        computed_entry.delete(0, tk.END)  # Clear the entry box
        computed_entry.insert(0, computed_result)  # Insert the computed value
        
    except ValueError:
        # If the entered value is not an integer, handle the error
        computed_entry.delete(0, tk.END)
        computed_entry.insert(0, "Invalid input")
        
'''    
def goal_socket(): 
	print("***Group goal*****!!!!!")
        global socket
        #udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = goal_entry.get()
        print(data,type(data),"KKKKK")
        d="goal"+","+str(data)
        print("d",d)
        '''
        udp_socket.sendto(str(d).encode(), server_address1)
        time.sleep(0.5)
        udp_socket2.sendto(str(d).encode(), server_address2)
        time.sleep(0.5)
        '''
        udp_socket3.sendto(str(d).encode(), server_address3)
        time.sleep(0.5)
        udp_socket4.sendto(str(d).encode(), server_address4)
        time.sleep(0.5)
        udp_socket5.sendto(str(d).encode(), server_address5)
        time.sleep(0.5)
        '''
        udp_socket6.sendto(str(d).encode(), server_address6)
        time.sleep(0.5)
        udp_socket7.sendto(str(d).encode(), server_address7)
        time.sleep(0.5)
        udp_socket8.sendto(str(d).encode(), server_address8)
        time.sleep(0.5)
        udp_socket9.sendto(str(d).encode(), server_address9)
        time.sleep(0.5)
        '''
        udp_socket10.sendto(str(d).encode(), server_address10)
        goal_entry.bind("<Return>",specific_bot_goal_socket)
        

def move_bot_socket(): 
	print("$$$#####")
        global socket
        #udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = move_bot_entry.get()
        print(data,type(data),"KKKKK")
        d="move_bot"+","+str(data)
        print("d",d)
        '''
        udp_socket.sendto(str(d).encode(), server_address1)
        time.sleep(0.5)
        udp_socket2.sendto(str(d).encode(), server_address2)
        time.sleep(0.5)
        '''
        udp_socket3.sendto(str(d).encode(), server_address3)
        time.sleep(0.5)
        udp_socket4.sendto(str(d).encode(), server_address4)
        time.sleep(0.5)
        udp_socket5.sendto(str(d).encode(), server_address5)
        time.sleep(0.5)
        '''
        udp_socket6.sendto(str(d).encode(), server_address6)
        time.sleep(0.5)
        udp_socket7.sendto(str(d).encode(), server_address7)
        time.sleep(0.5)
        udp_socket8.sendto(str(d).encode(), server_address8)
        time.sleep(0.5)
        udp_socket9.sendto(str(d).encode(), server_address9)
        time.sleep(0.5)
        '''
        udp_socket10.sendto(str(d).encode(), server_address10)
        move_bot_entry.bind("<Return>",move_bot_socket)


'''	
def enter():  
        print("^^^^^$$$$")
        socket1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = specific_bot_goal_entry.get()
        socket1.sendto(str(data).encode(),server_address11)
        specific_bot_goal_entry.bind("<Return>",enter)
'''        

       
 
def disconnect():
    #global serial_object
    try:
        print ("##serial_object.close()")
        ####serial_object.close() 
    
    except AttributeError:
        print ("Closed without Using it -_-")

    root.quit()



def mode_button1():
    global vehicle1
    print ("mode ok")
    print ("###########################################")
    mode=var1.get()
    print (mode)
    if str(mode) == 'STABILIZE':
        vehicle1.mode = VehicleMode("STABILIZE")
    elif str(mode) == 'LOITER':
        vehicle1.mode = VehicleMode("LOITER")
    elif str(mode) == 'AUTO':
        vehicle1.mode = VehicleMode("AUTO")
    elif str(mode) == 'GUIDED':
        vehicle1.mode = VehicleMode("GUIDED")
    elif str(mode) == 'RTL':
        vehicle1.mode = VehicleMode("RTL")
    elif str(mode) == 'LAND':
        vehicle1.mode = VehicleMode("LAND")
    
def mode_button2():
    print ("mode2 ok")
    global vehicle2
    mode=var2.get()
    if str(mode) == 'STABILIZE':
        vehicle2.mode = VehicleMode("STABILIZE")
    elif str(mode) == 'LOITER':
        vehicle2.mode = VehicleMode("LOITER")
    elif str(mode) == 'AUTO':
        vehicle2.mode = VehicleMode("AUTO")
    elif str(mode) == 'GUIDED':
        vehicle2.mode = VehicleMode("GUIDED")
    elif str(mode) == 'RTL':
        vehicle2.mode = VehicleMode("RTL")
    elif str(mode) == 'LAND':
        vehicle2.mode = VehicleMode("LAND")


def mode_button3():
    global vehicle3
    mode=var3.get()
    if str(mode) == 'STABILIZE':
        vehicle3.mode = VehicleMode("STABILIZE")
    elif str(mode) == 'LOITER':
        vehicle3.mode = VehicleMode("LOITER")
    elif str(mode) == 'AUTO':
        vehicle3.mode = VehicleMode("AUTO")
    elif str(mode) == 'GUIDED':
        vehicle3.mode = VehicleMode("GUIDED")
    elif str(mode) == 'RTL':
        vehicle3.mode = VehicleMode("RTL")
    elif str(mode) == 'LAND':
        vehicle3.mode = VehicleMode("LAND")

def mode_button4():
    global vehicle4
    mode=var4.get()
    if str(mode) == 'STABILIZE':
        vehicle4.mode = VehicleMode("STABILIZE")
    elif str(mode) == 'LOITER':
        vehicle4.mode = VehicleMode("LOITER")
    elif str(mode) == 'AUTO':
        vehicle4.mode = VehicleMode("AUTO")
    elif str(mode) == 'GUIDED':
        vehicle4.mode = VehicleMode("GUIDED")
    elif str(mode) == 'RTL':
        vehicle4.mode = VehicleMode("RTL")
    elif str(mode) == 'LAND':
        vehicle4.mode = VehicleMode("LAND")

def mode_button5():
    global vehicle5
    mode=var5.get()
    if str(mode) == 'STABILIZE':
        vehicle5.mode = VehicleMode("STABILIZE")
    elif str(mode) == 'LOITER':
        vehicle5.mode = VehicleMode("LOITER")
    elif str(mode) == 'AUTO':
        vehicle5.mode = VehicleMode("AUTO")
    elif str(mode) == 'GUIDED':
        vehicle5.mode = VehicleMode("GUIDED")
    elif str(mode) == 'RTL':
        vehicle5.mode = VehicleMode("RTL")
    elif str(mode) == 'LAND':
        vehicle5.mode = VehicleMode("LAND")

def mode_button6():
    global vehicle6
    mode=var6.get()
    if str(mode) == 'STABILIZE':
        vehicle6.mode = VehicleMode("STABILIZE")
    elif str(mode) == 'LOITER':
        vehicle6.mode = VehicleMode("LOITER")
    elif str(mode) == 'AUTO':
        vehicle6.mode = VehicleMode("AUTO")
    elif str(mode) == 'GUIDED':
        vehicle6.mode = VehicleMode("GUIDED")
    elif str(mode) == 'RTL':
        vehicle6.mode = VehicleMode("RTL")
    elif str(mode) == 'LAND':
        vehicle6.mode = VehicleMode("LAND")

def mode_button7():
    global vehicle7
    mode=var7.get()
    if str(mode) == 'STABILIZE':
        vehicle7.mode = VehicleMode("STABILIZE")
    elif str(mode) == 'LOITER':
        vehicle7.mode = VehicleMode("LOITER")
    elif str(mode) == 'AUTO':
        vehicle7.mode = VehicleMode("AUTO")
    elif str(mode) == 'GUIDED':
        vehicle7.mode = VehicleMode("GUIDED")
    elif str(mode) == 'RTL':
        vehicle7.mode = VehicleMode("RTL")
    elif str(mode) == 'LAND':
        vehicle7.mode = VehicleMode("LAND")

def mode_button8():
    global vehicle8
    mode=var8.get()
    if str(mode) == 'STABILIZE':
        vehicle8.mode = VehicleMode("STABILIZE")
    elif str(mode) == 'LOITER':
        vehicle8.mode = VehicleMode("LOITER")
    elif str(mode) == 'AUTO':
        vehicle8.mode = VehicleMode("AUTO")
    elif str(mode) == 'GUIDED':
        vehicle8.mode = VehicleMode("GUIDED")
    elif str(mode) == 'RTL':
        vehicle8.mode = VehicleMode("RTL")
    elif str(mode) == 'LAND':
        vehicle8.mode = VehicleMode("LAND")

def mode_button9():
    global vehicle9
    mode=var9.get()
    if str(mode) == 'STABILIZE':
        vehicle9.mode = VehicleMode("STABILIZE")
    elif str(mode) == 'LOITER':
        vehicle9.mode = VehicleMode("LOITER")
    elif str(mode) == 'AUTO':
        vehicle9.mode = VehicleMode("AUTO")
    elif str(mode) == 'GUIDED':
        vehicle9.mode = VehicleMode("GUIDED")
    elif str(mode) == 'RTL':
        vehicle9.mode = VehicleMode("RTL")
    elif str(mode) == 'LAND':
        vehicle9.mode = VehicleMode("LAND")

def mode_button10():
    global vehicle10
    mode=var10.get()
    if str(mode) == 'STABILIZE':
        vehicle10.mode = VehicleMode("STABILIZE")
    elif str(mode) == 'LOITER':
        vehicle10.mode = VehicleMode("LOITER")
    elif str(mode) == 'AUTO':
        vehicle10.mode = VehicleMode("AUTO")
    elif str(mode) == 'GUIDED':
        vehicle10.mode = VehicleMode("GUIDED")
    elif str(mode) == 'RTL':
        vehicle10.mode = VehicleMode("RTL")
    elif str(mode) == 'LAND':
        vehicle10.mode = VehicleMode("LAND")

def mode_button11():
    global vehicle11
    mode=var11.get()
    if str(mode) == 'STABILIZE':
        vehicle11.mode = VehicleMode("STABILIZE")
    elif str(mode) == 'LOITER':
        vehicle11.mode = VehicleMode("LOITER")
    elif str(mode) == 'AUTO':
        vehicle11.mode = VehicleMode("AUTO")
    elif str(mode) == 'GUIDED':
        vehicle11.mode = VehicleMode("GUIDED")
    elif str(mode) == 'RTL':
        vehicle11.mode = VehicleMode("RTL")
    elif str(mode) == 'LAND':
        vehicle11.mode = VehicleMode("LAND")

def mode_button12():
    global vehicle12
    mode=var12.get()
    if str(mode) == 'STABILIZE':
        vehicle12.mode = VehicleMode("STABILIZE")
    elif str(mode) == 'LOITER':
        vehicle12.mode = VehicleMode("LOITER")
    elif str(mode) == 'AUTO':
        vehicle12.mode = VehicleMode("AUTO")
    elif str(mode) == 'GUIDED':
        vehicle12.mode = VehicleMode("GUIDED")
    elif str(mode) == 'RTL':
        vehicle12.mode = VehicleMode("RTL")
    elif str(mode) == 'LAND':
        vehicle12.mode = VehicleMode("LAND")

def mode_button13():
    global vehicle13
    mode=var13.get()
    if str(mode) == 'STABILIZE':
        vehicle13.mode = VehicleMode("STABILIZE")
    elif str(mode) == 'LOITER':
        vehicle13.mode = VehicleMode("LOITER")
    elif str(mode) == 'AUTO':
        vehicle13.mode = VehicleMode("AUTO")
    elif str(mode) == 'GUIDED':
        vehicle13.mode = VehicleMode("GUIDED")
    elif str(mode) == 'RTL':
        vehicle13.mode = VehicleMode("RTL")
    elif str(mode) == 'LAND':
        vehicle13.mode = VehicleMode("LAND")

def mode_button14():
    global vehicle14
    mode=var14.get()
    if str(mode) == 'STABILIZE':
        vehicle14.mode = VehicleMode("STABILIZE")
    elif str(mode) == 'LOITER':
        vehicle14.mode = VehicleMode("LOITER")
    elif str(mode) == 'AUTO':
        vehicle14.mode = VehicleMode("AUTO")
    elif str(mode) == 'GUIDED':
        vehicle14.mode = VehicleMode("GUIDED")
    elif str(mode) == 'RTL':
        vehicle14.mode = VehicleMode("RTL")
    elif str(mode) == 'LAND':
        vehicle14.mode = VehicleMode("LAND")

def mode_button15():
    global vehicle15
    mode=var15.get()
    if str(mode) == 'STABILIZE':
        vehicle15.mode = VehicleMode("STABILIZE")
    elif str(mode) == 'LOITER':
        vehicle15.mode = VehicleMode("LOITER")
    elif str(mode) == 'AUTO':
        vehicle15.mode = VehicleMode("AUTO")
    elif str(mode) == 'GUIDED':
        vehicle15.mode = VehicleMode("GUIDED")
    elif str(mode) == 'RTL':
        vehicle15.mode = VehicleMode("RTL")
    elif str(mode) == 'LAND':
        vehicle15.mode = VehicleMode("LAND")

def mode_button16():
    global vehicle16
    mode=var16.get()
    if str(mode) == 'STABILIZE':
        vehicle16.mode = VehicleMode("STABILIZE")
    elif str(mode) == 'LOITER':
        vehicle16.mode = VehicleMode("LOITER")
    elif str(mode) == 'AUTO':
        vehicle16.mode = VehicleMode("AUTO")
    elif str(mode) == 'GUIDED':
        vehicle16.mode = VehicleMode("GUIDED")
    elif str(mode) == 'RTL':
        vehicle16.mode = VehicleMode("RTL")
    elif str(mode) == 'LAND':
        vehicle16.mode = VehicleMode("LAND")

    
def mode_button17():
    global vehicle17
    mode=var17.get()
    if str(mode) == 'STABILIZE':
        vehicle17.mode = VehicleMode("STABILIZE")
    elif str(mode) == 'LOITER':
        vehicle17.mode = VehicleMode("LOITER")
    elif str(mode) == 'AUTO':
        vehicle17.mode = VehicleMode("AUTO")
    elif str(mode) == 'GUIDED':
        vehicle17.mode = VehicleMode("GUIDED")
    elif str(mode) == 'RTL':
        vehicle17.mode = VehicleMode("RTL")
    elif str(mode) == 'LAND':
        vehicle17.mode = VehicleMode("LAND")

def mode_button18():
    global vehicle18
    mode=var18.get()
    if str(mode) == 'STABILIZE':
        vehicle18.mode = VehicleMode("STABILIZE")
    elif str(mode) == 'LOITER':
        vehicle18.mode = VehicleMode("LOITER")
    elif str(mode) == 'AUTO':
        vehicle18.mode = VehicleMode("AUTO")
    elif str(mode) == 'GUIDED':
        vehicle18.mode = VehicleMode("GUIDED")
    elif str(mode) == 'RTL':
        vehicle18.mode = VehicleMode("RTL")
    elif str(mode) == 'LAND':
        vehicle18.mode = VehicleMode("LAND")

def mode_button19():
    global vehicle19
    mode=var19.get()
    if str(mode) == 'STABILIZE':
        vehicle19.mode = VehicleMode("STABILIZE")
    elif str(mode) == 'LOITER':
        vehicle19.mode = VehicleMode("LOITER")
    elif str(mode) == 'AUTO':
        vehicle19.mode = VehicleMode("AUTO")
    elif str(mode) == 'GUIDED':
        vehicle19.mode = VehicleMode("GUIDED")
    elif str(mode) == 'RTL':
        vehicle19.mode = VehicleMode("RTL")
    elif str(mode) == 'LAND':
        vehicle19.mode = VehicleMode("LAND")

def mode_button20():
    global vehicle20
    mode=var20.get()
    if str(mode) == 'STABILIZE':
        vehicle20.mode = VehicleMode("STABILIZE")
    elif str(mode) == 'LOITER':
        vehicle20.mode = VehicleMode("LOITER")
    elif str(mode) == 'AUTO':
        vehicle20.mode = VehicleMode("AUTO")
    elif str(mode) == 'GUIDED':
        vehicle20.mode = VehicleMode("GUIDED")
    elif str(mode) == 'RTL':
        vehicle20.mode = VehicleMode("RTL")
    elif str(mode) == 'LAND':
        vehicle20.mode = VehicleMode("LAND")

def mode_button21():
    global vehicle21
    mode=var21.get()
    if str(mode) == 'STABILIZE':
        vehicle21.mode = VehicleMode("STABILIZE")
    elif str(mode) == 'LOITER':
        vehicle21.mode = VehicleMode("LOITER")
    elif str(mode) == 'AUTO':
        vehicle21.mode = VehicleMode("AUTO")
    elif str(mode) == 'GUIDED':
        vehicle21.mode = VehicleMode("GUIDED")
    elif str(mode) == 'RTL':
        vehicle21.mode = VehicleMode("RTL")
    elif str(mode) == 'LAND':
        vehicle21.mode = VehicleMode("LAND")

    
def mode_button22():
    global vehicle22
    mode=var22.get()
    if str(mode) == 'STABILIZE':
        vehicle22.mode = VehicleMode("STABILIZE")
    elif str(mode) == 'LOITER':
        vehicle22.mode = VehicleMode("LOITER")
    elif str(mode) == 'AUTO':
        vehicle22.mode = VehicleMode("AUTO")
    elif str(mode) == 'GUIDED':
        vehicle22.mode = VehicleMode("GUIDED")
    elif str(mode) == 'RTL':
        vehicle22.mode = VehicleMode("RTL")
    elif str(mode) == 'LAND':
        vehicle22.mode = VehicleMode("LAND")

def mode_button23():
    global vehicle23
    mode=var23.get()
    if str(mode) == 'STABILIZE':
        vehicle23.mode = VehicleMode("STABILIZE")
    elif str(mode) == 'LOITER':
        vehicle23.mode = VehicleMode("LOITER")
    elif str(mode) == 'AUTO':
        vehicle23.mode = VehicleMode("AUTO")
    elif str(mode) == 'GUIDED':
        vehicle23.mode = VehicleMode("GUIDED")
    elif str(mode) == 'RTL':
        vehicle23.mode = VehicleMode("RTL")
    elif str(mode) == 'LAND':
        vehicle23.mode = VehicleMode("LAND")

def mode_button24():
    global vehicle24
    mode=var24.get()
    if str(mode) == 'STABILIZE':
        vehicle24.mode = VehicleMode("STABILIZE")
    elif str(mode) == 'LOITER':
        vehicle24.mode = VehicleMode("LOITER")
    elif str(mode) == 'AUTO':
        vehicle24.mode = VehicleMode("AUTO")
    elif str(mode) == 'GUIDED':
        vehicle24.mode = VehicleMode("GUIDED")
    elif str(mode) == 'RTL':
        vehicle24.mode = VehicleMode("RTL")
    elif str(mode) == 'LAND':
        vehicle24.mode = VehicleMode("LAND")

def mode_button25():
    global vehicle25
    mode=var25.get()
    if str(mode) == 'STABILIZE':
        vehicle25.mode = VehicleMode("STABILIZE")
    elif str(mode) == 'LOITER':
        vehicle25.mode = VehicleMode("LOITER")
    elif str(mode) == 'AUTO':
        vehicle25.mode = VehicleMode("AUTO")
    elif str(mode) == 'GUIDED':
        vehicle25.mode = VehicleMode("GUIDED")
    elif str(mode) == 'RTL':
        vehicle25.mode = VehicleMode("RTL")
    elif str(mode) == 'LAND':
        vehicle25.mode = VehicleMode("LAND")

def mode_button_1():
    global follower_host_tuple
    print ("mode2 ok")
    """
    global vehicle2
    mode=var2.get()
    """
    for iter_follower in follower_host_tuple: 
	if iter_follower == None:
	    print ("slave is lost")
	else:
	    for i in range(0, 2):       
	    	iter_follower.airspeed = 10
	    	time.sleep(0.2)
    if checkboxvalue_1.get() == 1:
	goto_lat_p=s_lat1.get()
	goto_lon_p=s_lon1.get()

	goto_lat_p=float(goto_lat_p)
	goto_lon_p=float(goto_lon_p)
	formation_move(goto_lat_p, goto_lon_p, 50, 270) 
def mode_button_2():
    global follower_host_tuple
    """
    print ("mode2 ok")
    global vehicle2
    mode=var2.get()
    """
    for iter_follower in follower_host_tuple: 
	if iter_follower == None:
	    print ("slave is lost")
	else:
	    for i in range(0, 2):       
	    	iter_follower.airspeed = 10
	    	time.sleep(0.2)
    if checkboxvalue_2.get() == 1:
	goto_lat_p=s_lat2.get()
	goto_lon_p=s_lon2.get()

	goto_lat_p=float(goto_lat_p)
	goto_lon_p=float(goto_lon_p)
	formation_move(goto_lat_p, goto_lon_p, 50, 270) 
def mode_button_3():
    global follower_host_tuple
    """
    print ("mode2 ok")
    global vehicle2
    mode=var2.get()
    """
    for iter_follower in follower_host_tuple: 
	if iter_follower == None:
	    print ("slave is lost")
	else:
	    for i in range(0, 2):       
	    	iter_follower.airspeed = 10
	    	time.sleep(0.2)
    if checkboxvalue_3.get() == 1:
	goto_lat_p=s_lat3.get()
	goto_lon_p=s_lon3.get()

	goto_lat_p=float(goto_lat_p)
	goto_lon_p=float(goto_lon_p)
	formation_move(goto_lat_p, goto_lon_p, 50, 270) 
def mode_button_4():
    global follower_host_tuple
    """
    print ("mode2 ok")
    global vehicle2
    mode=var2.get()
    """
    for iter_follower in follower_host_tuple: 
	if iter_follower == None:
	    print ("slave is lost")
	else:
	    for i in range(0, 2):       
	    	iter_follower.airspeed = 10
	    	time.sleep(0.2)
    if checkboxvalue_4.get() == 1:
	goto_lat_p=s_lat4.get()
	goto_lon_p=s_lon4.get()

	goto_lat_p=float(goto_lat_p)
	goto_lon_p=float(goto_lon_p)
	formation_move(goto_lat_p, goto_lon_p, 50, 270) 
def mode_button_5():
    """
    print ("mode2 ok")
    global vehicle2
    mode=var2.get()
    """
def mode_button_6():
    print ("mode2 ok")
    global vehicle2
    mode=var2.get()
def mode_button_7():
    print ("mode2 ok")
    global vehicle2
    mode=var2.get()
def mode_button_8():
    print ("mode2 ok")
    global vehicle2
    mode=var2.get()
def mode_button_9():
    print ("mode2 ok")
    global vehicle2
    mode=var2.get()
def mode_button_10():
    print ("mode2 ok")
    global vehicle2
    mode=var2.get()
def mode_button_11():
    print ("mode2 ok")
    global vehicle2
    mode=var2.get()
def mode_button_12():
    print ("mode2 ok")
    global vehicle2
    mode=var2.get()
def mode_button_13():
    print ("mode2 ok")
    global vehicle2
    mode=var2.get()
def mode_button_14():
    print ("mode2 ok")
    global vehicle2
    mode=var2.get()
def mode_button_15():
    print ("mode2 ok")
    global vehicle2
    mode=var2.get()
def mode_button_16():
    print ("mode2 ok")
    global vehicle2
    mode=var2.get()
def mode_button_17():
    print ("mode2 ok")
    global vehicle2
    mode=var2.get()
def mode_button_18():
    print ("mode2 ok")
    global vehicle2
    mode=var2.get()
def mode_button_19():
    print ("mode2 ok")
    global vehicle2
    mode=var2.get()
def mode_button_20():
    print ("mode2 ok")
    global vehicle2
    mode=var2.get()
def mode_button_21():
    print ("mode2 ok")
    global vehicle2
    mode=var2.get()
def mode_button_22():
    print ("mode2 ok")
    global vehicle2
    mode=var2.get()
def mode_button_23():
    print ("mode2 ok")
    global vehicle2
    mode=var2.get()
def mode_button_24():
    print ("mode2 ok")
    global vehicle2
    mode=var2.get()
def mode_button_25():
    print ("mode2 ok")
    global vehicle2
    mode=var2.get()



def goto_button1():
    global vehicle1
    goto_lat=g_lat1.get()
    goto_lon=g_lon1.get()

    goto_lat=float(goto_lat)
    goto_lon=float(goto_lon)

    goto_location(vehicle1, goto_lat,goto_lon)
def goto_button2():
    global vehicle2
    goto_lat=g_lat2.get()
    goto_lon=g_lon2.get()

    goto_lat=float(goto_lat)
    goto_lon=float(goto_lon)

    goto_location(vehicle2, goto_lat,goto_lon)
def goto_button3():
    global vehicle3
    goto_lat=g_lat3.get()
    goto_lon=g_lon3.get()

    goto_lat=float(goto_lat)
    goto_lon=float(goto_lon)

    goto_location(vehicle3, goto_lat,goto_lon)
def goto_button4():
    global vehicle4
    goto_lat=g_lat4.get()
    goto_lon=g_lon4.get()

    goto_lat=float(goto_lat)
    goto_lon=float(goto_lon)

    goto_location(vehicle4, goto_lat,goto_lon)
def goto_button5():
    global vehicle5
    goto_lat=g_lat5.get()
    goto_lon=g_lon5.get()

    goto_lat=float(goto_lat)
    goto_lon=float(goto_lon)

    goto_location(vehicle5, goto_lat,goto_lon)
def goto_button6():
    global vehicle6
    goto_lat=g_lat6.get()
    goto_lon=g_lon6.get()

    goto_lat=float(goto_lat)
    goto_lon=float(goto_lon)

    goto_location(vehicle6, goto_lat,goto_lon)
def goto_button7():
    global vehicle7
    goto_lat=g_lat7.get()
    goto_lon=g_lon7.get()

    goto_lat=float(goto_lat)
    goto_lon=float(goto_lon)

    goto_location(vehicle7, goto_lat,goto_lon)
def goto_button8():
    global vehicle8
    goto_lat=g_lat8.get()
    goto_lon=g_lon8.get()

    goto_lat=float(goto_lat)
    goto_lon=float(goto_lon)

    goto_location(vehicle8, goto_lat,goto_lon)
def goto_button9():
    global vehicle9
    goto_lat=g_lat9.get()
    goto_lon=g_lon9.get()

    goto_lat=float(goto_lat)
    goto_lon=float(goto_lon)

    goto_location(vehicle9, goto_lat,goto_lon)
def goto_button10():
    global vehicle10
    goto_lat=g_lat10.get()
    goto_lon=g_lon10.get()

    goto_lat=float(goto_lat)
    goto_lon=float(goto_lon)

    goto_location(vehicle10, goto_lat,goto_lon)
def goto_button11():
    global vehicle11
    goto_lat=g_lat11.get()
    goto_lon=g_lon11.get()
    goto_alt=g_alt11.get()
    goto_lat=float(goto_lat)
    goto_lon=float(goto_lon)
    goto_alt=int(goto_alt)
    goto_location(vehicle11, goto_lat,goto_lon)
def goto_button12():
    global vehicle12
    goto_lat=g_lat12.get()
    goto_lon=g_lon12.get()

    goto_lat=float(goto_lat)
    goto_lon=float(goto_lon)

    goto_location(vehicle12, goto_lat,goto_lon)
def goto_button13():
    global vehicle13
    goto_lat=g_lat13.get()
    goto_lon=g_lon13.get()

    goto_lat=float(goto_lat)
    goto_lon=float(goto_lon)

    goto_location(vehicle13, goto_lat,goto_lon)
def goto_button14():
    global vehicle14
    goto_lat=g_lat14.get()
    goto_lon=g_lon14.get()

    goto_lat=float(goto_lat)
    goto_lon=float(goto_lon)

    goto_location(vehicle14, goto_lat,goto_lon)
def goto_button15():
    global vehicle15
    goto_lat=g_lat15.get()
    goto_lon=g_lon15.get()

    goto_lat=float(goto_lat)
    goto_lon=float(goto_lon)

    goto_location(vehicle15, goto_lat,goto_lon)
def goto_button16():
    global vehicle16
    goto_lat=g_lat16.get()
    goto_lon=g_lon16.get()

    goto_lat=float(goto_lat)
    goto_lon=float(goto_lon)

    goto_location(vehicle16, goto_lat,goto_lon)
def goto_button17():
    global vehicle17
    goto_lat=g_lat17.get()
    goto_lon=g_lon17.get()

    goto_lat=float(goto_lat)
    goto_lon=float(goto_lon)

    goto_location(vehicle17, goto_lat,goto_lon)
def goto_button18():
    global vehicle18
    goto_lat=g_lat18.get()
    goto_lon=g_lon18.get()

    goto_lat=float(goto_lat)
    goto_lon=float(goto_lon)

    goto_location(vehicle18, goto_lat,goto_lon)
def goto_button19():
    global vehicle19
    goto_lat=g_lat19.get()
    goto_lon=g_lon19.get()

    goto_lat=float(goto_lat)
    goto_lon=float(goto_lon)

    goto_location(vehicle19, goto_lat,goto_lon,goto_alt)
def goto_button20():
    global vehicle20
    goto_lat=g_lat20.get()
    goto_lon=g_lon20.get()

    goto_lat=float(goto_lat)
    goto_lon=float(goto_lon)

    goto_location(vehicle20, goto_lat,goto_lon)
def goto_button21():
    global vehicle21
    goto_lat=g_lat21.get()
    goto_lon=g_lon21.get()

    goto_lat=float(goto_lat)
    goto_lon=float(goto_lon)

    goto_location(vehicle21, goto_lat,goto_lon)
def goto_button22():
    global vehicle22
    goto_lat=g_lat22.get()
    goto_lon=g_lon22.get()

    goto_lat=float(goto_lat)
    goto_lon=float(goto_lon)

    goto_location(vehicle22, goto_lat,goto_lon)
def goto_button23():
    global vehicle23
    goto_lat=g_lat23.get()
    goto_lon=g_lon23.get()

    goto_lat=float(goto_lat)
    goto_lon=float(goto_lon)

    goto_location(vehicle23, goto_lat,goto_lon)
def goto_button24():
    global vehicle24
    goto_lat=g_lat24.get()
    goto_lon=g_lon24.get()

    goto_lat=float(goto_lat)
    goto_lon=float(goto_lon)

    goto_location(vehicle24, goto_lat,goto_lon)
def goto_button25():
    global vehicle25
    goto_lat=g_lat25.get()
    goto_lon=g_lon25.get()

    goto_lat=float(goto_lat)
    goto_lon=float(goto_lon)

    goto_location(vehicle25, goto_lat,goto_lon)

def alt_button1():
    global vehicle1
    goto_alt=g_alt1.get()
    goto_alt=int(goto_alt)
    altitude_inc(vehicle1, goto_alt)

def alt_button2():
    global vehicle2
    goto_alt=g_alt2.get()
    goto_alt=int(goto_alt)
    altitude_inc(vehicle2, goto_alt)

def alt_button3():
    global vehicle3
    goto_alt=g_alt3.get()
    goto_alt=int(goto_alt)
    altitude_inc(vehicle3, goto_alt)

def alt_button4():
    global vehicle4
    goto_alt=g_alt4.get()
    goto_alt=int(goto_alt)
    altitude_inc(vehicle4, goto_alt)

def alt_button5():
    global vehicle5
    goto_alt=g_alt5.get()
    goto_alt=int(goto_alt)
    altitude_inc(vehicle5, goto_alt)

def alt_button6():
    global vehicle6
    goto_alt=g_alt6.get()
    goto_alt=int(goto_alt)
    altitude_inc(vehicle6, goto_alt)

def alt_button7():
    global vehicle7
    goto_alt=g_alt7.get()
    goto_alt=int(goto_alt)
    altitude_inc(vehicle7, goto_alt)

def alt_button8():
    global vehicle8
    goto_alt=g_alt8.get()
    goto_alt=int(goto_alt)
    altitude_inc(vehicle8, goto_alt)

def alt_button9():
    global vehicle9
    goto_alt=g_alt9.get()
    goto_alt=int(goto_alt)
    altitude_inc(vehicle9, goto_alt)

def alt_button10():
    global vehicle10
    goto_alt=g_alt10.get()
    goto_alt=int(goto_alt)
    altitude_inc(vehicle10, goto_alt)

def alt_button11():
    global vehicle11
    goto_alt=g_alt11.get()
    goto_alt=int(goto_alt)
    altitude_inc(vehicle11, goto_alt)

def alt_button12():
    global vehicle12
    goto_alt=g_alt12.get()
    goto_alt=int(goto_alt)
    altitude_inc(vehicle12, goto_alt)

def alt_button13():
    global vehicle13
    goto_alt=g_alt13.get()
    goto_alt=int(goto_alt)
    altitude_inc(vehicle13, goto_alt)

def alt_button14():
    global vehicle14
    goto_alt=g_alt14.get()
    goto_alt=int(goto_alt)
    altitude_inc(vehicle14, goto_alt)

def alt_button15():
    global vehicle15
    goto_alt=g_alt15.get()
    goto_alt=int(goto_alt)
    altitude_inc(vehicle15, goto_alt)

def alt_button16():
    global vehicle16
    goto_alt=g_alt16.get()
    goto_alt=int(goto_alt)
    altitude_inc(vehicle16, goto_alt)

def alt_button17():
    global vehicle17
    goto_alt=g_alt17.get()
    goto_alt=int(goto_alt)
    altitude_inc(vehicle17, goto_alt)

def alt_button18():
    global vehicle18
    goto_alt=g_alt18.get()
    goto_alt=int(goto_alt)
    altitude_inc(vehicle18, goto_alt)

def alt_button19():
    global vehicle19
    goto_alt=g_alt19.get()
    goto_alt=int(goto_alt)
    altitude_inc(vehicle19, goto_alt)

def alt_button20():
    global vehicle20
    goto_alt=g_alt20.get()
    goto_alt=int(goto_alt)
    altitude_inc(vehicle20, goto_alt)

def alt_button21():
    global vehicle21
    goto_alt=g_alt21.get()
    goto_alt=int(goto_alt)
    altitude_inc(vehicle21, goto_alt)

def alt_button22():
    global vehicle22
    goto_alt=g_alt22.get()
    goto_alt=int(goto_alt)
    altitude_inc(vehicle22, goto_alt)

def alt_button23():
    global vehicle23
    goto_alt=g_alt23.get()
    goto_alt=int(goto_alt)
    altitude_inc(vehicle23, goto_alt)

def alt_button24():
    global vehicle24
    goto_alt=g_alt24.get()
    goto_alt=int(goto_alt)
    altitude_inc(vehicle24, goto_alt)

def alt_button25():
    global vehicle25
    goto_alt=g_alt25.get()
    goto_alt=int(goto_alt)
    altitude_inc(vehicle25, goto_alt)



def payload_button1():
    global vehicle1
    payload_send(vehicle1)
def payload_button2():
    global vehicle2
    payload_send(vehicle2)
def payload_button3():
    global vehicle3
    payload_send(vehicle3)
def payload_button4():
    global vehicle4
    payload_send(vehicle4)
def payload_button5():
    global vehicle5
    payload_send(vehicle5)
def payload_button6():
    global vehicle6
    payload_send(vehicle6)
def payload_button7():
    global vehicle7
    payload_send(vehicle7)
def payload_button8():
    global vehicle8
    payload_send(vehicle8)
def payload_button9():
    global vehicle9
    payload_send(vehicle9)
def payload_button10():
    global vehicle10
    payload_send(vehicle10)
def payload_button11():
    global vehicle11
    payload_send(vehicle11)
def payload_button12():
    global vehicle12
    payload_send(vehicle12)
def payload_button13():
    global vehicle13
    payload_send(vehicle13)
def payload_button14():
    global vehicle14
    payload_send(vehicle14)
def payload_button15():
    global vehicle15
    payload_send(vehicle15)
def payload_button16():
    global vehicle16
    payload_send(vehicle16)
def payload_button17():
    global vehicle17
    payload_send(vehicle17)
def payload_button18():
    global vehicle18
    payload_send(vehicle18)
def payload_button19():
    global vehicle19
    payload_send(vehicle19)
def payload_button20():
    global vehicle20
    payload_send(vehicle20)
def payload_button21():
    global vehicle21
    payload_send(vehicle21)
def payload_button22():
    global vehicle22
    payload_send(vehicle22)
def payload_button23():
    global vehicle23
    payload_send(vehicle23)
def payload_button24():
    global vehicle24
    payload_send(vehicle24)
def payload_button25():
    global vehicle25
    payload_send(vehicle25)


def get_location_metres(lat, lon, dEast,dNorth):
    earth_radius = 6378137.0 #Radius of "spherical" earth
    #Coordinate offsets in radians
    dLat = dNorth/earth_radius
    dLon = dEast/(earth_radius*math.cos(math.pi*lat/180))

    #New position in decimal degrees
    newlat = lat + (dLat * 180/math.pi)
    newlon = lon + (dLon * 180/math.pi)
  
    targetlat, targetlon = (newlat, newlon)
    return targetlat, targetlon


def altered_position(original_location,dEast,dNorth):
   print ("original_location,dEast,dNorth", original_location,dEast,dNorth)
   earth_radius = 6378137.0
   dLat = dNorth/earth_radius
   dLon = dEast/(earth_radius*cos(pi*original_location[0]/180))
   newlat = original_location[0] + (dLat *180/pi)
   newlon = original_location[1] + (dLon *180/pi)
   targetlocation = (newlat,newlon)
   return targetlocation

"""
def get_location_metres(original_location, dNorth, dEast):
    dEast,dNorth = int(dEast),int(dNorth)
    earth_radius = 6378137#Radius of "spherical" earth
    #Coordinate offsets in radians
    dLat = dNorth/earth_radius
    dLon = dEast/(earth_radius*math.cos(math.pi*original_location.lat/180))

    #New position in decimal degrees
    newlat = original_location.lat + (dLat * 180/math.pi)
    newlon = original_location.lon + (dLon * 180/math.pi)
    if type(original_location) is LocationGlobal:
        targetlocation=LocationGlobal(newlat, newlon,original_location.alt)
    elif type(original_location) is LocationGlobalRelative:
        targetlocation=LocationGlobalRelative(newlat, newlon,original_location.alt)
    else:
        raise Exception("Invalid Location object passed")
        
    return targetlocation;
"""
'''
def payload_drop_all_one_point():
    global self_heal, master
    global all_pos_data, xy_pos, latlon_pos
    global check_box_flag4
    global circle_pos_flag
    global follower_host_tuple
    global no_uavs

    #if circle_pos_flag == True:
    clat = clat_entry.get()
    clon = clon_entry.get()
    clat = float(clat)
    clon = float(clon)
    #alt_123 = 50
    
    if master == 1:
	goto_lat=g_lat1.get()
	goto_lon=g_lon1.get()

	lat=float(goto_lat)
	lon=float(goto_lon)
    if master == 2:
	goto_lat=g_lat2.get()
	goto_lon=g_lon2.get()

	lat=float(goto_lat)
	lon=float(goto_lon)

    if master == 3:
	goto_lat=g_lat3.get()
	goto_lon=g_lon3.get()

	lat=float(goto_lat)
	lon=float(goto_lon)

    if master == 4:
	goto_lat=g_lat4.get()
	goto_lon=g_lon4.get()

	lat=float(goto_lat)
	lon=float(goto_lon)

    formation_heading(no_uavs, 'L', lat, lon, 270)
    alt_12345 = 50
    for i, iter_follower in enumerate(follower_host_tuple):  
    	if self_heal[i] > 0:
		print ("slave is lost")
		#alt_123 = alt_123+10
	else:  
		print ("goto_drop location")
		lat_origin = iter_follower.location.global_frame.lat
		lon_origin = iter_follower.location.global_frame.lon
		alt_origin = iter_follower.location.global_frame.alt
		goto_location_drop(iter_follower, clat, clon, 'Forward', i)
	        print ("...goto_location_drop_forward....")
		#goto_location_drop(iter_follower, lat_origin, lon_origin, 'Return')
		test = latlon_pos[i]
                
		goto_location_line(iter_follower, test[0], test[1], alt_12345)
		alt_12345 = alt_12345+10
		#goto_location(iter_follower, test[0], test[1])
		print (".....goto_location_drop_return....")
'''

def aggr_line():
    global self_heal, master
    global all_pos_data, xy_pos, latlon_pos
    global check_box_flag4
    global circle_pos_flag
    global follower_host_tuple
    global no_uavs
    for i, iter_follower in enumerate(follower_host_tuple):  
    	if self_heal[i] > 0:
		print ("slave is lost")
		#alt_123 = alt_123+10
	else:  
		test = latlon_pos[i]
		#goto_location_line(iter_follower, test[0], test[1], alt_123)
		goto_location(iter_follower, test[0], test[1])
		print (".....goto_location....")



def goto_location_drop(vehicle, latitude, longitude, mode_data, array_pos):
    global check_box_flag4
    global self_heal
    lat = vehicle.location.global_relative_frame.lat
    lon = vehicle.location.global_relative_frame.lon
    alt = vehicle.location.global_relative_frame.alt
    alt = abs(alt)
    print "Navigating to point"
    target = LocationGlobalRelative(latitude, longitude, 30)
    #target = LocationGlobalRelative(latitude, longitude, vehicle.location.global_relative_frame.alt)
    print target
    min_distance = 1 # Parameter to tune by experimentin
    start = time.time()
    print ("....array pos ......", array_pos)
    for i in range(0, 3):
    	if check_box_flag4 == True:
    		vehicle.simple_goto(target)
    while True:
        if check_box_flag4 == False:
		print ("loop is break")
		threading.Thread(target=air_break,args=(vehicle,)).start()
		#air_break(vehicle)
		break        
        print "target.lat, target.lon", (target.lat, target.lon)
        print "vehicle.lat, vehicle.lon", (vehicle.location.global_frame.lat, vehicle.location.global_frame.lon)
        lat1 = radians(vehicle.location.global_frame.lat)
        lon1 = radians(vehicle.location.global_frame.lon)
        lat2 = radians(target.lat)
        lon2 = radians(target.lon)
        dlon = lon2 - lon1
	dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
	c = 2 * atan2(sqrt(a), sqrt(1 - a))

	distance = R * c
        distance = (distance * 1000)
        print("Result:", distance)
        output = "dist b/w vehicle to geo_search and vehicle lat & lon:", (distance, lat, lon)
        output = str(output)
        ##ser.write(output+ '\r\n')
        if distance<=min_distance:
            print "Reached target location"
	    if mode_data == 'Forward':
		payload_send(vehicle)
	    if mode_data == 'Return':
		print ("return to origin")
            break;
        time.sleep(0.5)

def Air_break():
    global follower_host_tuple
    global self_heal

    for k, iter_follower in enumerate(follower_host_tuple): 
    	if self_heal[k] > 0:
    		print ("lost odd uav", (k+1))
    	else:
		print ("present uav")  
		threading.Thread(target=air_break,args=(iter_follower,)).start()
		#air_break(iter_follower)
		time.sleep(0.2)
		
def Target_Heading_compute():
    global self_heal  #move_all
    global master, no_uavs
    global xy_pos, latlon_pos
    global self_heal, self_heal
    global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10,vehicle11,vehicle12,vehicle13
    global vehicle14,vehicle15,vehicle16,vehicle17,vehicle18,vehicle19,vehicle20,vehicle21,vehicle22,vehicle23,vehicle24,vehicle25      
    global follower_host_tuple_G1,follower_host_tuple_G2,follower_host_tuple_G3,follower_host_tuple_G4,follower_host_tuple_G5
    print ("compute heading value and update real heading")
    #..................all...................
    if checkboxvalue_Group_all.get() == 1:

	    if master == 1:
		lat1 = vehicle1.location.global_relative_frame.lat
		lon1 = vehicle1.location.global_relative_frame.lon
		goto_lat=g_lat1.get()
		goto_lon=g_lon1.get()
		print ("goto_lat,goto_lon", goto_lat,goto_lon)
		try:
			tlat2=float(goto_lat)
			tlon2=float(goto_lon)
		except:
			pass	
		print ("tlat2,tlat2", tlat2,tlon2)			
	    if master == 2:
		lat1 = vehicle2.location.global_relative_frame.lon
		lon1 = vehicle2.location.global_relative_frame.alt
		#alt = 110

		goto_lat=g_lat2.get()
		goto_lon=g_lon2.get()
		try:
			tlat2=float(goto_lat)
			tlon2=float(goto_lon)
		except:
			pass
	    if master == 3:
		lat1 = vehicle3.location.global_relative_frame.lat
		lon1 = vehicle3.location.global_relative_frame.lon
		#alt = 120

		goto_lat=g_lat3.get()
		goto_lon=g_lon3.get()
		try:
			tlat1=float(goto_lat)
			tlon2=float(goto_lon)
		except:
			pass
	    try:
	    	print ("lat1, lon1, tlat1, tlon2", lat1, lon1, tlat2, tlon2)
	    	target_heading = gps_bearing(lat1, lon1, tlat2, tlon2)
		print ("...target_heading...", target_heading)
	    	#..THD.delete(0, END) 


	    	THD.insert(0, str(abs(target_heading)))
	    	print ("...ok...")
	    except:
	    	pass
	    
	    

def move_payload():
    global self_heal  #move_all
    global master, no_uavs
    global xy_pos, latlon_pos
    global self_heal, self_heal
    global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10,vehicle11,vehicle12,vehicle13
    global vehicle14,vehicle15     
    global follower_host_tuple_G1,follower_host_tuple_G2,follower_host_tuple_G3,follower_host_tuple_G4,follower_host_tuple_G5
    global counter_G1,counter_G2,counter_G3,counter_G4,counter_G5
    global follower_host_tuple
    global circle_pos_flag
    xoffset = xoffset_entry.get()    
    cradius = cradius_entry.get()    
    aoffset = aoffset_entry.get()    
    salt = salt_entry.get() 
    xoffset = int(xoffset)    
    cradius = int(cradius)    
    aoffset = int(aoffset)
    salt = int(salt)
    print ("aggregation", master)


    #..................all...................
    if checkboxvalue_Group_all.get() == 1:

	    if master == 1:
		#lat = vehicle1.location.global_relative_frame.lat
		#lon = vehicle1.location.global_relative_frame.lon
		goto_lat=g_lat1.get()
		goto_lon=g_lon1.get()

		lat=float(goto_lat)
		lon=float(goto_lon)
	    if master == 2:
		#lon = vehicle2.location.global_relative_frame.lon
		##alt = vehicle2.location.global_relative_frame.alt
		#alt = 110

		goto_lat=g_lat2.get()
		goto_lon=g_lon2.get()

		lat=float(goto_lat)
		lon=float(goto_lon)

	    if master == 3:
		#lat = vehicle3.location.global_relative_frame.lat
		#lon = vehicle3.location.global_relative_frame.lon
		#alt = 120

		goto_lat=g_lat3.get()
		goto_lon=g_lon3.get()

		lat=float(goto_lat)
		lon=float(goto_lon)

	    if master == 4:
		#lat = vehicle4.location.global_relative_frame.lat
		#lon = vehicle4.location.global_relative_frame.lon
		#alt = 130

		goto_lat=g_lat4.get()
		goto_lon=g_lon4.get()

		lat=float(goto_lat)
		lon=float(goto_lon)

	    if master == 5:
		#lat = vehicle5.location.global_relative_frame.lat
		#lon = vehicle5.location.global_relative_frame.lon
		#alt = 140

		goto_lat=g_lat5.get()
		goto_lon=g_lon5.get()

		lat=float(goto_lat)
		lon=float(goto_lon)

	    if master == 6:
		#lat = vehicle6.location.global_relative_frame.lat
		#lon = vehicle6.location.global_relative_frame.lon
		#alt = 150

		goto_lat=g_lat6.get()
		goto_lon=g_lon6.get()

		lat=float(goto_lat)
		lon=float(goto_lon)

	    if master == 7:
		#lat = vehicle7.location.global_relative_frame.lat
		#lon = vehicle7.location.global_relative_frame.lon
		#alt = 160

		goto_lat=g_lat7.get()
		goto_lon=g_lon7.get()

		lat=float(goto_lat)
		lon=float(goto_lon)

	    if master == 8:
		#lat = vehicle8.location.global_relative_frame.lat
		#lon = vehicle8.location.global_relative_frame.lon
		#alt = 170

		goto_lat=g_lat8.get()
		goto_lon=g_lon8.get()

		lat=float(goto_lat)
		lon=float(goto_lon)

	    if master == 9:
		#lat = vehicle9.location.global_relative_frame.lat
		#lon = vehicle9.location.global_relative_frame.lon
		#alt = 180

		goto_lat=g_lat9.get()
		goto_lon=g_lon9.get()

		lat=float(goto_lat)
		lon=float(goto_lon)

	    if master == 10:
		#lat = vehicle10.location.global_relative_frame.lat
		#lon = vehicle10.location.global_relative_frame.lon
		#alt = 190

		goto_lat=g_lat10.get()
		goto_lon=g_lon10.get()

		lat=float(goto_lat)
		lon=float(goto_lon)
	    


	    if checkboxvalue1.get() == 1:
		formation(int(no_uavs), 'T', lat, lon)
	    elif checkboxvalue2.get() == 1:
		formation(int(no_uavs), 'L', lat, lon)
	    elif checkboxvalue3.get() == 1:
		formation(int(no_uavs), 'S', lat, lon)
	    elif checkboxvalue4.get() == 1:
		if circle_pos_flag == True:
		    circle_pos_flag = False
		    clat = clat_entry.get()
		    clon = clon_entry.get()
		    clat = float(clat)
		    clon = float(clon)
		    formation(int(no_uavs), 'C', clat, clon)
		else:
		    formation(int(no_uavs), 'C', lat, lon)


	    a,b,c = (0,0,0)
	    count_wp = 0
	    alt_000 = salt

	    for i, iter_follower in enumerate(follower_host_tuple): 
		#i = i+1
		if self_heal[i] > 0:
		    ##print "lost odd uav", self_heal[i]
		    print ("lost odd uav", (i+1))
		    pos_latlon = (0.0, 0.0)
		    latlon_pos.insert(i, (0.0, 0.0))
		    ##c = (c+20)   
		    if check_box_flag3 == True: 
		        print ("self heal..to alt change")   
		    else:
		        alt_000 = alt_000 + aoffset 
		else:   
		    test = latlon_pos[i]
		    print ("..t_goto..", test[0], test[1])   
		    if i == 0:    
		    	goto_alt1=g_alt1.get()
		    	goto_alt1=int(goto_alt1) 
		    	target = LocationGlobalRelative(test[0], test[1], goto_alt1)
		    if i == 1:    
		    	goto_alt2=g_alt2.get()
		    	goto_alt2=int(goto_alt2) 
		    	target = LocationGlobalRelative(test[0], test[1], goto_alt2)
		    if i == 2:    
		    	goto_alt3=g_alt3.get()
		    	goto_alt3=int(goto_alt3) 
		    	target = LocationGlobalRelative(test[0], test[1], goto_alt3)
		    if i == 3:    
		    	goto_alt4=g_alt4.get()
		    	goto_alt4=int(goto_alt4) 
		    	target = LocationGlobalRelative(test[0], test[1], goto_alt4)
		    if i == 4:    
		    	goto_alt5=g_alt5.get()
		    	goto_alt5=int(goto_alt5) 
		    	target = LocationGlobalRelative(test[0], test[1], goto_alt5)
		    if i == 5:    
		    	goto_alt6=g_alt6.get()
		    	goto_alt6=int(goto_alt6) 
		    	target = LocationGlobalRelative(test[0], test[1], goto_alt6)
		    if i == 6:    
		    	goto_alt7=g_alt7.get()
		    	goto_alt7=int(goto_alt7) 
		    	target = LocationGlobalRelative(test[0], test[1], goto_alt7)
		    if i == 7:    
		    	goto_alt8=g_alt8.get()
		    	goto_alt8=int(goto_alt8) 
		    	target = LocationGlobalRelative(test[0], test[1], goto_alt8)
		    if i == 8:    
		    	goto_alt9=g_alt9.get()
		    	goto_alt9=int(goto_alt9) 
		    	target = LocationGlobalRelative(test[0], test[1], goto_alt9)
		    if i == 9:    
		    	goto_alt10=g_alt10.get()
		    	goto_alt10=int(goto_alt10) 
		    	target = LocationGlobalRelative(test[0], test[1], goto_alt10)

		    print ("target", target)
		    aggregation_formation(iter_follower,"GUIDED", target)        
		    alt_000 = alt_000 + aoffset
	    #...................end..................


def vehicle_collision_moniter():
    global follower_host_tuple
    global check_box_flag5
    while True:
	time.sleep(0.2)
	loc_1 = []
	for val_1 in range(0, 6):
		location_1 = (follower_host_tuple[val_1].location.global_frame.lat, follower_host_tuple[val_1].location.global_frame.lon)
		loc_1.append(location_1)

	for current_1 in range(1, len(loc_1)+1):
		for each_1 in range(1, len(loc_1)+1):
			if current_1 != each_1:
				distance_1, bearing_1 = locate.distance_bearing(loc_1[current_1-1][0], loc_1[current_1-1][1], loc_1[each_1-1][0], loc_1[each_1-1][1])			
				#print ("distance", distance_1)
				if distance_1 < 10:
					print (".....!!!!!!!!!!!!!...........collision......uav,dist", current_1, distance_1)
					##Air_break()


def payload_drop_vehicle1_moniter():
    global check_box_flag5
    global self_heal
    clat = clat_entry.get()
    clon = clon_entry.get()
    clat = float(clat)
    clon = float(clon)
    if self_heal[0] > 0:
	print ("vehicle1 is lost")
    else:
    	target_p = LocationGlobalRelative(clat, clon, vehicle1.location.global_relative_frame.alt)
    	min_distance = 4
    while True:
    	if self_heal[0] > 0:
		print ("vehicle1 is lost") 
		time.sleep(0.5)   
    	else:    
		lat1 = radians(vehicle1.location.global_frame.lat)
		lon1 = radians(vehicle1.location.global_frame.lon)
		lat2 = radians(target_p.lat)
		lon2 = radians(target_p.lon)
		dlon = lon2 - lon1
		dlat = lat2 - lat1
		a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
		c = 2 * atan2(sqrt(a), sqrt(1 - a))

		distance_1 = R * c
		distance_1 = (distance_1 * 1000)
		#print("dist_uav1:", distance_1)
		if distance_1<=min_distance:
		    print "********UAV1 Reached target location********"
		    payload_send(vehicle1)
		    break;
		if check_box_flag5 == False:
		    print "*******break1********"
		    break;

		time.sleep(0.5)
def payload_drop_vehicle2_moniter():
    global check_box_flag5
    global self_heal
    clat = clat_entry.get()
    clon = clon_entry.get()
    clat = float(clat)
    clon = float(clon)
    if self_heal[1] > 0:
	print ("vehicle2 is lost")
    else:
    	target_p = LocationGlobalRelative(clat, clon, vehicle2.location.global_relative_frame.alt)
    	min_distance = 4
    while True:       
    	if self_heal[1] > 0:
		print ("vehicle2 is lost")  
		time.sleep(0.5)  
    	else:
		lat1 = radians(vehicle2.location.global_frame.lat)
		lon1 = radians(vehicle2.location.global_frame.lon)
		lat2 = radians(target_p.lat)
		lon2 = radians(target_p.lon)
		dlon = lon2 - lon1
		dlat = lat2 - lat1
		a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
		c = 2 * atan2(sqrt(a), sqrt(1 - a))

		distance_2 = R * c
		distance_2 = (distance_2 * 1000)
		#print("dist_uav2:", distance_2)
		if distance_2<=min_distance:
		    print "********UAV2 Reached target location********"
		    payload_send(vehicle2)
		    break;
		if check_box_flag5 == False:
		    print "*******break2********"
		    break;
		time.sleep(0.5)
def payload_drop_vehicle3_moniter():
    global check_box_flag5
    global self_heal
    clat = clat_entry.get()
    clon = clon_entry.get()
    clat = float(clat)
    clon = float(clon)
    if self_heal[2] > 0:
	print ("vehicle3 is lost")
    else:
    	target_p = LocationGlobalRelative(clat, clon, vehicle3.location.global_relative_frame.alt)
    	min_distance = 4
    while True:   
    	if self_heal[2] > 0:
		print ("vehicle3 is lost") 
		time.sleep(0.5)   
    	else:    
		lat1 = radians(vehicle3.location.global_frame.lat)
		lon1 = radians(vehicle3.location.global_frame.lon)
		lat2 = radians(target_p.lat)
		lon2 = radians(target_p.lon)
		dlon = lon2 - lon1
		dlat = lat2 - lat1
		a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
		c = 2 * atan2(sqrt(a), sqrt(1 - a))

		distance_3 = R * c
		distance_3 = (distance_3 * 1000)
		#print("dist_uav3:", distance_3)
		if distance_3<=min_distance:
		    print "********UAV3 Reached target location********"
		    payload_send(vehicle3)
		    break;
		if check_box_flag5 == False:
		    print "*******break3********"
		    break;
		time.sleep(0.5)
def payload_drop_vehicle4_moniter():
    global check_box_flag5
    global self_heal
    clat = clat_entry.get()
    clon = clon_entry.get()
    clat = float(clat)
    clon = float(clon)
    if self_heal[3] > 0:
	print ("vehicle4 is lost")
    else:
    	target_p = LocationGlobalRelative(clat, clon, vehicle4.location.global_relative_frame.alt)
    	min_distance = 4
    while True:       
    	if self_heal[3] > 0:
		print ("vehicle4 is lost") 
		time.sleep(0.5)   
    	else:
		lat1 = radians(vehicle4.location.global_frame.lat)
		lon1 = radians(vehicle4.location.global_frame.lon)
		lat2 = radians(target_p.lat)
		lon2 = radians(target_p.lon)
		dlon = lon2 - lon1
		dlat = lat2 - lat1
		a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
		c = 2 * atan2(sqrt(a), sqrt(1 - a))

		distance_4 = R * c
		distance_4 = (distance_4 * 1000)
		#print("dist_uav4:", distance_4)
		if distance_4<=min_distance:
		    print "********UAV4 Reached target location********"
		    payload_send(vehicle4)
		    break;
		if check_box_flag5 == False:
		    print "*******break4********"
		    break;
		time.sleep(0.5)
def payload_drop_vehicle5_moniter():
    global check_box_flag5
    global self_heal
    clat = clat_entry.get()
    clon = clon_entry.get()
    clat = float(clat)
    clon = float(clon)
    if self_heal[4] > 0:
	print ("vehicle5 is lost")
    else:
    	target_p = LocationGlobalRelative(clat, clon, vehicle5.location.global_relative_frame.alt)
    	min_distance = 4
    while True:   
    	if self_heal[4] > 0:
		print ("vehicle5 is lost")   
		time.sleep(0.5) 
    	else:    
		lat1 = radians(vehicle5.location.global_frame.lat)
		lon1 = radians(vehicle5.location.global_frame.lon)
		lat2 = radians(target_p.lat)
		lon2 = radians(target_p.lon)
		dlon = lon2 - lon1
		dlat = lat2 - lat1
		a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
		c = 2 * atan2(sqrt(a), sqrt(1 - a))

		distance_5 = R * c
		distance_5 = (distance_5 * 1000)
		#print("dist_uav5:", distance_5)

		if distance_5<=min_distance:
		    print "********UAV5 Reached target location********"
		    payload_send(vehicle5)
		    break;
		if check_box_flag5 == False:
		    print "*******break5********"
		    break;
		time.sleep(0.5)
def payload_drop_vehicle6_moniter():
    global check_box_flag5
    global self_heal
    clat = clat_entry.get()
    clon = clon_entry.get()
    clat = float(clat)
    clon = float(clon)
    if self_heal[5] > 0:
	print ("vehicle6 is lost")
    else:
    	target_p = LocationGlobalRelative(clat, clon, vehicle6.location.global_relative_frame.alt)
    	min_distance = 4
    while True:   
    	if self_heal[5] > 0:
		print ("vehicle6 is lost")  
		time.sleep(0.5) 
    	else:    
		lat1 = radians(vehicle6.location.global_frame.lat)
		lon1 = radians(vehicle6.location.global_frame.lon)
		lat2 = radians(target_p.lat)
		lon2 = radians(target_p.lon)
		dlon = lon2 - lon1
		dlat = lat2 - lat1
		a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
		c = 2 * atan2(sqrt(a), sqrt(1 - a))

		distance_6 = R * c
		distance_6 = (distance_6 * 1000)
		#print("dist_uav6:", distance_6)
		if distance_6<=min_distance:
		    print "********UAV6 Reached target location********"
		    payload_send(vehicle6)
		    break;
		if check_box_flag5 == False:
		    print "*******break6********"
		    break;
		time.sleep(0.5)



def goto_location_line(vehicle, goto_lat, goto_lon, alt_1234):

    #target=LocationGlobalRelative(goto_lat, goto_lon, alt_1234)
    target=LocationGlobalRelative(goto_lat, goto_lon, alt_1234)
    timeout = 100
    vehicle.mode = VehicleMode("GUIDED")
    time.sleep(1)
    min_distance = 0.000005 # Parameter to tune by experimenting
    for i in range(0, 3):
    	vehicle.simple_goto(target)

def goto_location(vehicle, goto_lat, goto_lon):

    target=LocationGlobalRelative(goto_lat, goto_lon, vehicle.location.global_relative_frame.alt)
    timeout = 100
    vehicle.mode = VehicleMode("GUIDED")
    time.sleep(1)
    min_distance = 0.000005 # Parameter to tune by experimenting
    for i in range(0, 3):
    	vehicle.simple_goto(target)
    """
    start = time.time()    
    while vehicle.mode.name=="GUIDED":
    current = time.time() - start
    dTarget = sqrt(pow(target.lat-vehicle.location.global_frame.lat,2)+pow(target.lon-vehicle.location.global_frame.lon,2)++pow(target.alt-vehicle.location.global_relative_frame.alt,2))
    print " ->T:%0.1f, Target[%0.2f %0.2f %0.1f], Actual[%0.2f %0.2f %0.1f], ToGo:%0.6f" % (current, target.lat, target.lon, target.alt, vehicle.location.global_frame.lat, vehicle.location.global_frame.lon, vehicle.location.global_relative_frame.alt, dTarget)
    if dTarget<=min_distance:
        print "Reached target location"
        break;nt compl
    if current >= timeout:
        print "Timeout to reach location, last distance: %0.4f" % (dTarget)
        break;
    time.sleep(0.5)
     """


def point_pos(x0, y0, d, theta):
    theta_rad = pi/2 - radians(theta)
    return x0 + d*cos(theta_rad), y0 + d*sin(theta_rad)

def getPlaneCoordinates(lat, lon):
    x, y, z = transform(wgs84,fleast_m,lon,lat,0.0)
    return x*M2FT,y*M2FT

def getGeoCoordinates(x, y):
    lat, lon, depth = transform(fleast_m,wgs84,x*FT2M,y*FT2M,0.0)
    return lon, lat
    
def gps_bearing(lat1, lon1, lat2, lon2):
    print (",g,g,,g,")
    '''return bearing between two points in degrees, in range 0-360
    thanks to http://www.movable-type.co.uk/scripts/latlong.html'''
    from math import sin, cos, atan2, radians, degrees
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    dLat = lat2 - lat1
    dLon = lon2 - lon1    
    y = sin(dLon) * cos(lat2)
    x = cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(dLon)
    bearing = degrees(atan2(y, x))
    print ("....")
    if bearing < 0:
        bearing += 360.0
    return bearing
    
    
def gps_distance(lat1, lon1, lat2, lon2):
    '''return distance between two points in meters,
    coordinates are in degrees
    thanks to http://www.movable-type.co.uk/scripts/latlong.html'''
    from math import radians, cos, sin, sqrt, atan2
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    dLat = lat2 - lat1
    dLon = lon2 - lon1
    
    a = sin(0.5*dLat)**2 + sin(0.5*dLon)**2 * cos(lat1) * cos(lat2)
    c = 2.0 * atan2(sqrt(a), sqrt(1.0-a))
    return radius_of_earth * c
    
    
def gps_distance(lat1, lon1, lat2, lon2):
    '''return distance between two points in meters,
    coordinates are in degrees
    thanks to http://www.movable-type.co.uk/scripts/latlong.html'''
    from math import radians, cos, sin, sqrt, atan2
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    dLat = lat2 - lat1
    dLon = lon2 - lon1
    
    a = sin(0.5*dLat)**2 + sin(0.5*dLon)**2 * cos(lat1) * cos(lat2)
    c = 2.0 * atan2(sqrt(a), sqrt(1.0-a))
    return radius_of_earth * c


def gps_newpos(lat, lon, bearing, distance, t_alt):
    '''extrapolate latitude/longitude given a heading and distance 
    thanks to http://www.movable-type.co.uk/scripts/latlong.html
    '''
    from math import sin, asin, cos, atan2, radians, degrees

    lat1 = radians(lat)
    lon1 = radians(lon)
    brng = radians(bearing)
    dr = distance/radius_of_earth

    lat2 = asin(sin(lat1)*cos(dr) +
            cos(lat1)*sin(dr)*cos(brng))
    lon2 = lon1 + atan2(sin(brng)*sin(dr)*cos(lat1), 
                cos(dr)-sin(lat1)*sin(lat2))
    return (degrees(lat2), degrees(lon2), t_alt)


def search_loc(lat,lon, dist_between_waypoints, heading, num_rows, num_cols, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4,ln_param7,alt_slave):
    global missionlist
    half_dist = dist_between_waypoints/2
    x,y = getPlaneCoordinates(lat, lon)
    #print "mmmmmmmmmmmmmmmmmm"
    print ("x,y",(x,y))

    for i in range(0,num_cols):
           col_offset = half_dist*(2*i+1)
           x1, y1 = point_pos(x, y, col_offset, (heading+90)%360)
           
           for j in range(0,num_rows):
                if i % 2:
                   row_offset = half_dist*((num_rows-j)*2-2)
                else:
                   row_offset = half_dist*(2*j)
                x2, y2 = point_pos(x1, y1, row_offset, heading)
                lat2, lon2 = getGeoCoordinates(x2, y2)
                # print (lat2, lon2)
                #tar_data= [float(lat2), float(lon2)]
                if i == 0:
                        if j == 0:
                            print ("1st row starting pt alt increase")         
                            #loc.append(tar_data)
                            targetLocation = LocationGlobalRelative(lat2, lon2, ln_param7)
                            cmd = Command( 0, 0, 0, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4, targetLocation.lat, targetLocation.lon, targetLocation.alt)

                            missionlist.append(cmd)
                else:
                        if j == 0:
                            print ("all row remaining starting point")                  
                            targetLocation = LocationGlobalRelative(lat2, lon2, alt_slave)
                            cmd = Command( 0, 0, 0, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4, targetLocation.lat, targetLocation.lon, targetLocation.alt)

                            missionlist.append(cmd)
                if i == 0:
                        if j == 1:  #alt 35
                            print ("2 pt")

                            targetLocation = LocationGlobalRelative(lat2, lon2, alt_slave)
                            cmd = Command( 0, 0, 0, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4, targetLocation.lat, targetLocation.lon, targetLocation.alt)

                            missionlist.append(cmd)

                if i == num_cols-1:  
                        if j == num_rows-2:  # alt 35
                            print ("last previus pt")

                            targetLocation = LocationGlobalRelative(lat2, lon2, alt_slave)
                            cmd = Command( 0, 0, 0, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4, targetLocation.lat, targetLocation.lon, targetLocation.alt)

                            missionlist.append(cmd)



                if i == num_cols-1:
                    if j == num_rows-1:
                        print ("last point alt inrease")

                        targetLocation = LocationGlobalRelative(lat2, lon2, ln_param7)
                        cmd = Command( 0, 0, 0, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4, targetLocation.lat, targetLocation.lon, targetLocation.alt)

                        missionlist.append(cmd)
                else:
                        if j == num_rows-1:
                            print ("all row end point")

                            targetLocation = LocationGlobalRelative(lat2, lon2, alt_slave)

                            
                            cmd = Command( 0, 0, 0, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4, targetLocation.lat, targetLocation.lon, targetLocation.alt)

                            missionlist.append(cmd)

def render(poly):
	"""Return polygon as grid of points inside polygon.

	Input : poly (list of lists)
	Output : output (list of lists)
	"""
	xs, ys = zip(*poly)
	minx, maxx = min(xs), max(xs)
	miny, maxy = min(ys), max(ys)

	newPoly = [(int(x - minx), int(y - miny)) for (x, y) in poly]

	X = maxx - minx + 1
	Y = maxy - miny + 1

	grid = np.zeros((X, Y), dtype=np.int8)
	mahotas.polygon.fill_polygon(newPoly, grid)

	return [(x + minx, y + miny) for (x, y) in zip(*np.nonzero(grid))]

def reverse(seq, start, stop):
	size = stop + start
	for i in range(start, (size + 1) // 2 ):
		j = size - i
		seq[i], seq[j] = seq[j], seq[i]



def readmission(vehicle,aFileName,a,b,c,number,model):
   global missionlist_uav_all
   global missionlist, home_locations
   global missionlist_uav1,missionlist_uav2,missionlist_uav3,missionlist_uav4,missionlist_uav5,missionlist_uav6,missionlist_uav7,missionlist_uav8,missionlist_uav9,missionlist_uav10
   global missionlist_uav11,missionlist_uav12,missionlist_uav13,missionlist_uav14,missionlist_uav15,missionlist_uav16,missionlist_uav17,missionlist_uav18,missionlist_uav19,missionlist_uav20
   global missionlist_uav21,missionlist_uav22,missionlist_uav23,missionlist_uav24,missionlist_uav25
   global latlon_pos
   if vehicle == None:
        print ("slave is lost")
   else:
    print("\nReading mission from file: %s" % aFileName)
    print ("waypoint_aggregation")
    missionlist=[]
    missionlist0 = []  #uav search start"
    missionlist_uav1 = []
    missionlist_uav2 = [] 
    missionlist_uav3 = []
    missionlist_uav4 = []
    missionlist_uav5 = []
    missionlist_uav6 = []
    missionlist_uav7 = []
    missionlist_uav8 = []
    missionlist_uav9 = []
    missionlist_uav10 = []
    missionlist_uav11 = []
    missionlist_uav12 = []
    missionlist_uav13 = []
    missionlist_uav14 = []
    missionlist_uav15 = []
    missionlist_uav16 = []
    missionlist_uav17 = []
    missionlist_uav18 = []
    missionlist_uav19 = []
    missionlist_uav20 = []
    missionlist_uav21 = []
    missionlist_uav22 = []
    missionlist_uav23 = []
    missionlist_uav24 = []
    missionlist_uav25 = []
    #missionlist_uav_all = []
    with open(aFileName) as f:
        for i, line in enumerate(f):
            i = i+1
        print ("*******len_wp*****", i-2)
        last_wp = (i-2)
    with open(aFileName) as f:
        for i, line in enumerate(f): 
            if i==0:
                if not line.startswith('QGC WPL 110'):
                    raise Exception('File is not supported WP version')
            elif i==1:
                    print ("first way point reject")
                
            elif (i) == last_wp:
                #print ("************last_prev_wp")
                        
                linearray=line.split('\t')
                ln_index=int(linearray[0])
                ln_currentwp=int(linearray[1])
                ln_frame=int(linearray[2])
                ln_command=int(linearray[3])
                ln_param1=float(linearray[4])
                ln_param2=float(linearray[5])
                ln_param3=float(linearray[6])
                ln_param4=float(linearray[7])
                ln_param5=float(linearray[8])
                ln_param6=float(linearray[9])
                ln_param7=float(linearray[10])
                ln_autocontinue=int(linearray[11].strip())

                original_location = [ln_param5, ln_param6]
                if checkboxvalue1.get() == 1:
                    formation(int(no_uavs), 'T', ln_param5, ln_param6)
                elif checkboxvalue2.get() == 1:
                    formation(int(no_uavs), 'L', ln_param5, ln_param6)
                elif checkboxvalue3.get() == 1:
                    formation(int(no_uavs), 'S', ln_param5, ln_param6)

                #init_pos1 = altered_position(original_location,0,0)
                test = latlon_pos[number-1]
                ln_param7=float(linearray[10])+c

                ln_param5 = test[0]
                ln_param6 = test[1]                     
    
                ln_autocontinue=int(linearray[11].strip())         
                cmd = Command( 0, 0, 0, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4, ln_param5, ln_param6, ln_param7)
                missionlist.append(cmd)
		if number == 1:
			missionlist_uav1.append(cmd)
		elif number == 2:
			missionlist_uav2.append(cmd)
		elif number == 3:
			missionlist_uav3.append(cmd)
		elif number == 4:
			missionlist_uav4.append(cmd)
		elif number == 5:
			missionlist_uav5.append(cmd)
		elif number == 6:
			missionlist_uav6.append(cmd)
		elif number == 7:
			missionlist_uav7.append(cmd)
		elif number == 8:
			missionlist_uav8.append(cmd)
		elif number == 9:
			missionlist_uav9.append(cmd)
		elif number == 10:
			missionlist_uav10.append(cmd)
		elif number == 11:
			missionlist_uav11.append(cmd)
		elif number == 12:
			missionlist_uav12.append(cmd)
		elif number == 13:
			missionlist_uav13.append(cmd)
		elif number == 14:
			missionlist_uav14.append(cmd)
		elif number == 15:
			missionlist_uav15.append(cmd)
		elif number == 16:
			missionlist_uav16.append(cmd)
		elif number == 17:
			missionlist_uav17.append(cmd)
		elif number == 18:
			missionlist_uav18.append(cmd)
		elif number == 19:
			missionlist_uav19.append(cmd)
		elif number == 20:
			missionlist_uav20.append(cmd)
		elif number == 21:
			missionlist_uav21.append(cmd)
		elif number == 22:
			missionlist_uav22.append(cmd)
		elif number == 23:
			missionlist_uav23.append(cmd)
		elif number == 24:
			missionlist_uav24.append(cmd)
		elif number == 25:
			missionlist_uav25.append(cmd)

            
            elif (i-1) == last_wp:
                #print ("************last_wp")
                linearray=line.split('\t')
                ln_index=int(linearray[0])
                ln_currentwp=int(linearray[1])
                ln_frame=int(linearray[2])
                ln_command=int(linearray[3])
                ln_param1=float(linearray[4])
                ln_param2=float(linearray[5])
                ln_param3=float(linearray[6])
                ln_param4=float(linearray[7])
                ln_param5=float(linearray[8])
                ln_param6=float(linearray[9])
                ln_param7=float(linearray[10])
                ln_autocontinue=int(linearray[11].strip())
                

                #print ("sssssssssssssssssss", number)
                h_loc = home_location[number-1]
                original_location = [h_loc[0], h_loc[1]]
                #original_location = [ln_param5, ln_param6]
                if checkboxvalue1.get() == 1:
                    formation(int(no_uavs), 'T', h_loc[0], h_loc[1])
                elif checkboxvalue2.get() == 1:
                    formation(int(no_uavs), 'L', h_loc[0], h_loc[1])
                elif checkboxvalue3.get() == 1:
                    formation(int(no_uavs), 'S', h_loc[0], h_loc[1])

                #init_pos1 = altered_position(original_location,0,0)
                test = latlon_pos[number-1]
                    
                ln_param7=float(linearray[10])+c

                ln_param5 = test[0]
                ln_param6 = test[1]       
                ln_autocontinue=int(linearray[11].strip())         
                cmd = Command( 0, 0, 0, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4, ln_param5, ln_param6, ln_param7)
                missionlist.append(cmd)
		if number == 1:
			missionlist_uav1.append(cmd)
		elif number == 2:
			missionlist_uav2.append(cmd)
		elif number == 3:
			missionlist_uav3.append(cmd)
		elif number == 4:
			missionlist_uav4.append(cmd)
		elif number == 5:
			missionlist_uav5.append(cmd)
		elif number == 6:
			missionlist_uav6.append(cmd)
		elif number == 7:
			missionlist_uav7.append(cmd)
		elif number == 8:
			missionlist_uav8.append(cmd)
		elif number == 9:
			missionlist_uav9.append(cmd)
		elif number == 10:
			missionlist_uav10.append(cmd)
		elif number == 11:
			missionlist_uav11.append(cmd)
		elif number == 12:
			missionlist_uav12.append(cmd)
		elif number == 13:
			missionlist_uav13.append(cmd)
		elif number == 14:
			missionlist_uav14.append(cmd)
		elif number == 15:
			missionlist_uav15.append(cmd)
		elif number == 16:
			missionlist_uav16.append(cmd)
		elif number == 17:
			missionlist_uav17.append(cmd)
		elif number == 18:
			missionlist_uav18.append(cmd)
		elif number == 19:
			missionlist_uav19.append(cmd)
		elif number == 20:
			missionlist_uav20.append(cmd)
		elif number == 21:
			missionlist_uav21.append(cmd)
		elif number == 22:
			missionlist_uav22.append(cmd)
		elif number == 23:
			missionlist_uav23.append(cmd)
		elif number == 24:
			missionlist_uav24.append(cmd)
		elif number == 25:
			missionlist_uav25.append(cmd)

            else:
                    
                    linearray=line.split('\t')
                    ln_index=int(linearray[0])
                    ln_currentwp=int(linearray[1])
                    ln_frame=int(linearray[2])
                    ln_command=int(linearray[3])
                    ln_param1=float(linearray[4])
                    ln_param2=float(linearray[5])
                    ln_param3=float(linearray[6])
                    ln_param4=float(linearray[7])
                    ln_param5=float(linearray[8])
                    ln_param6=float(linearray[9])
                    ln_param7=float(linearray[10])
                    ln_autocontinue=int(linearray[11].strip())
                    original_location = [ln_param5, ln_param6]

                        #speed_change
                    if ln_param1 == 10:
                    #if i >=4 and i<=6:
                        print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Target search area")
                        print ("model", model)
                        #.......................real search part 1........................
                        """
                        (c1lat, c1lon) = (12.930116, 80.046987)  #corner point1 # 
                        (c2lat, c2lon) = (12.93051, 80.04604)  #corner point1
                        (c3lat, c3lon) = (12.930147, 80.045901)  #corner point1
                        (c4lat, c4lon) = (12.929755, 80.046885)  #corner point1
                       
                        """
                        #.......................sim search part 1........................
                        
                        (c1lat, c1lon) = (12.9296355, 80.0466764)  #corner point1 # 
                        (c2lat, c2lon) = (12.9299178, 80.0456786)  #corner point1
                        (c3lat, c3lon) = (12.9308589, 80.0460112)  #corner point1
                        (c4lat, c4lon) = (12.9305034, 80.0470626)  #corner point1
                        #................................................
                        
                        #lat1, lon1, t_alt = gps_newpos(lat, lon, 45, 700, ln_param7+c)  #  sq(150)^2xsq(100)^2---->(300x200)- triangle of c side
                        #....lat1, lon1, t_alt = gps_newpos(lat, lon, 45, 300, ln_param7+c)  #  sq(150)^2xsq(100)^2---->(300x200)- triangle of c side
                        #ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4 
                        #0		0		3	16		0		0	0	0
                        UAV1_plat1,UAV1_plon1 = c1lat, c1lon  
                        UAV1_plat2,UAV1_plon2 = c2lat, c2lon   
                        target_bearing1 = gps_bearing(UAV1_plat1,UAV1_plon1, UAV1_plat2,UAV1_plon2)  
                        target_distance1 = gps_distance(UAV1_plat1,UAV1_plon1, UAV1_plat2,UAV1_plon2)     
                    
                        target_bearing2 = gps_bearing(UAV1_plat1,UAV1_plon1, c4lat, c4lon)  
                        target_distance2 = gps_distance(UAV1_plat1,UAV1_plon1, c4lat, c4lon)   
                    
                    
                        print ("UAV1_plat1,UAV1_plon1", UAV1_plat1,UAV1_plon1)   
                        print ("UAV1_plat2,UAV1_plon2", UAV1_plat2,UAV1_plon2)  
                        print ("target_bearing1", target_bearing1)
                        print ("target_distance1", target_distance1) 
                        print ("target_bearing2", target_bearing2)
                        print ("target_distance2", target_distance2)   
                        
                        #.......................real search part 2........................
                        """
                        (p1lat, p1lon) = (12.930093, 80.04704)  #corner point1
                        (p2lat, p2lon) = (12.92969, 80.048036)  #corner point1
                        (p3lat, p3lon) = (12.92932, 80.047894)  #corner point1
                        (p4lat, p4lon) = (12.929713, 80.04696)  #corner point1
                        """
                        
                        #.......................sim search part 2........................
                        
                        (p1lat, p1lon) = (12.8629, 79.9630)  #corner point1
                        (p2lat, p2lon) = (12.8636, 79.9638)  #corner point1
                        (p3lat, p3lon) = (12.891881610219485, 80.1217376633032)  #corner point1
                        (p4lat, p4lon) = (12.891885892494825, 80.12020898383781)  #corner point1
                        #..........................................
                        #lat1, lon1, t_alt = gps_newpos(lat, lon, 45, 700, ln_param7+c)  #  sq(150)^2xsq(100)^2---->(300x200)- triangle of c side
                        #....lat1, lon1, t_alt = gps_newpos(lat, lon, 45, 300, ln_param7+c)  #  sq(150)^2xsq(100)^2---->(300x200)- triangle of c side
                        #ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4 
                        #0		0		3	16		0		0	0	0
                        UAV2_plat1,UAV2_plon1 = p1lat, p1lon  
                        UAV2_plat2,UAV2_plon2 = p2lat, p2lon   
                        target_bearing1_2 = gps_bearing(UAV2_plat1,UAV2_plon1, UAV2_plat2,UAV2_plon2)  
                        target_distance1_2 = gps_distance(UAV2_plat1,UAV2_plon1, UAV2_plat2,UAV2_plon2)     
                    
                        target_bearing2_2 = gps_bearing(UAV2_plat1,UAV2_plon1, p4lat, p4lon)  
                        target_distance2_2 = gps_distance(UAV2_plat1,UAV2_plon1, p4lat, p4lon)   
                    
                    
                        print ("UAV2_plat1,UAV2_plon1", UAV2_plat1,UAV2_plon1)   
                        print ("UAV2_plat2,UAV2_plon2", UAV2_plat2,UAV2_plon2)  
                        print ("target_bearing1_2", target_bearing1_2)
                        print ("target_distance1_2", target_distance1_2) 
                        print ("target_bearing2_2", target_bearing2_2)
                        print ("target_distance2_2", target_distance2_2)   
                        #.......................real search part 3........................
                        """
                        (p31lat, p31lon) = (12.930093, 80.04704)  #corner point1
                        (p32lat, p32lon) = (12.92969, 80.048036)  #corner point1
                        (p33lat, p33lon) = (12.92932, 80.047894)  #corner point1
                        (p34lat, p34lon) = (12.929713, 80.04696)  #corner point1
                        """
                        
                        #.......................sim search part 3........................
                        
                        (p31lat, p31lon) = (12.8632, 79.9626)  #corner point1
                        (p32lat, p32lon) = (12.8642, 79.9621)  #corner point1
                        (p33lat, p33lon) = (12.891601697279727, 80.12172030107857)  #corner point1
                        (p34lat, p34lon) = (12.891617628260354, 80.12020211735582)  #corner point1
                        #..........................................
                        #lat1, lon1, t_alt = gps_newpos(lat, lon, 45, 700, ln_param7+c)  #  sq(150)^2xsq(100)^2---->(300x200)- triangle of c side
                        #....lat1, lon1, t_alt = gps_newpos(lat, lon, 45, 300, ln_param7+c)  #  sq(150)^2xsq(100)^2---->(300x200)- triangle of c side
                        #ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4 
                        #0		0		3	16		0		0	0	0
                        UAV3_plat1,UAV3_plon1 = p31lat, p31lon  
                        UAV3_plat2,UAV3_plon2 = p32lat, p32lon   
                        target_bearing1_3 = gps_bearing(UAV3_plat1,UAV3_plon1, UAV3_plat2,UAV3_plon2)  
                        target_distance1_3 = gps_distance(UAV3_plat1,UAV3_plon1, UAV3_plat2,UAV3_plon2)     
                    
                        target_bearing2_3 = gps_bearing(UAV3_plat1,UAV3_plon1, p34lat, p34lon)  
                        target_distance2_3 = gps_distance(UAV3_plat1,UAV3_plon1, p34lat, p34lon)   
                    
                    
                        print ("UAV3_plat1,UAV3_plon1", UAV3_plat1,UAV3_plon1)   
                        print ("UAV3_plat2,UAV3_plon2", UAV3_plat2,UAV3_plon2)  
                        print ("target_bearing1_3", target_bearing1_3)
                        print ("target_distance1_3", target_distance1_3) 
                        print ("target_bearing2_3", target_bearing2_3)
                        print ("target_distance2_3", target_distance2_3)   
                        #.......................real search part 4........................
                        """
                        (p1lat, p1lon) = (12.930093, 80.04704)  #corner point1
                        (p2lat, p2lon) = (12.92969, 80.048036)  #corner point1
                        (p3lat, p3lon) = (12.92932, 80.047894)  #corner point1
                        (p4lat, p4lon) = (12.929713, 80.04696)  #corner point1
                        """
                        
                        #.......................sim search part 4........................
                        
                        (p41lat, p41lon) = (12.8622, 79.9617)  #corner point1
                        (p42lat, p42lon) = (12.8615, 79.9601)  #corner point1
                        (p43lat, p43lon) = (12.891320229303885, 80.12181338054067)  #corner point1
                        (p44lat, p44lon) = (12.891327423452454, 80.12019081084333)  #corner point1
                        #..........................................
                        #lat1, lon1, t_alt = gps_newpos(lat, lon, 45, 700, ln_param7+c)  #  sq(150)^2xsq(100)^2---->(300x200)- triangle of c side
                        #....lat1, lon1, t_alt = gps_newpos(lat, lon, 45, 300, ln_param7+c)  #  sq(150)^2xsq(100)^2---->(300x200)- triangle of c side
                        #ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4 
                        #0		0		3	16		0		0	0	0
                        UAV4_plat1,UAV4_plon1 = p41lat, p41lon  
                        UAV4_plat2,UAV4_plon2 = p42lat, p42lon   
                        target_bearing1_4 = gps_bearing(UAV4_plat1,UAV4_plon1, UAV4_plat2,UAV4_plon2)  
                        target_distance1_4 = gps_distance(UAV4_plat1,UAV4_plon1, UAV4_plat2,UAV4_plon2)     
                    
                        target_bearing2_4 = gps_bearing(UAV4_plat1,UAV4_plon1, p44lat, p44lon)  
                        target_distance2_4 = gps_distance(UAV4_plat1,UAV4_plon1, p44lat, p44lon)   
                    
                    
                        print ("UAV4_plat1,UAV4_plon1", UAV4_plat1,UAV4_plon1)   
                        print ("UAV4_plat2,UAV4_plon2", UAV4_plat2,UAV4_plon2)  
                        print ("target_bearing1_4", target_bearing1_4)
                        print ("target_distance1_4", target_distance1_4) 
                        print ("target_bearing2_4", target_bearing2_4)
                        print ("target_distance2_4", target_distance2_4)   
                                                
                        #.......................sim search part 5........................
                        
                        (p51lat, p51lon) = (12.8609, 79.9699)  #corner point1
                        (p52lat, p52lon) = (12.8607, 79.9681)  #corner point1
                        (p53lat, p53lon) = (12.891320229303885, 80.12181338054067)  #corner point1
                        (p54lat, p54lon) = (12.891327423452454, 80.12019081084333)  #corner point1
                        #..........................................
                        #lat1, lon1, t_alt = gps_newpos(lat, lon, 45, 700, ln_param7+c)  #  sq(150)^2xsq(100)^2---->(300x200)- triangle of c side
                        #....lat1, lon1, t_alt = gps_newpos(lat, lon, 45, 300, ln_param7+c)  #  sq(150)^2xsq(100)^2---->(300x200)- triangle of c side
                        #ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4 
                        #0		0		3	16		0		0	0	0
                        UAV5_plat1,UAV5_plon1 = p51lat, p51lon  
                        UAV5_plat2,UAV5_plon2 = p52lat, p52lon   
                        target_bearing1_5 = gps_bearing(UAV5_plat1,UAV5_plon1, UAV5_plat2,UAV5_plon2)  
                        target_distance1_5 = gps_distance(UAV5_plat1,UAV5_plon1, UAV5_plat2,UAV5_plon2)     
                    
                        target_bearing2_5 = gps_bearing(UAV5_plat1,UAV5_plon1, p54lat, p54lon)  
                        target_distance2_5 = gps_distance(UAV5_plat1,UAV5_plon1, p54lat, p54lon)   
                    
                    
                        print ("UAV5_plat1,UAV5_plon1", UAV5_plat1,UAV5_plon1)   
                        print ("UAV5_plat2,UAV5_plon2", UAV5_plat2,UAV5_plon2)  
                        print ("target_bearing1_5", target_bearing1_5)
                        print ("target_distance1_5", target_distance1_5) 
                        print ("target_bearing2_5", target_bearing2_5)
                        print ("target_distance2_5", target_distance2_5)   
                                                
                        #.......................sim search part 6........................
                        
                        (p61lat, p61lon) = (12.8603, 79.9615)  #corner point1
                        (p62lat, p62lon) = (12.8613, 79.9628)  #corner point1
                        (p63lat, p63lon) = (12.891320229303885, 80.12181338054067)  #corner point1
                        (p64lat, p64lon) = (12.891327423452454, 80.12019081084333)  #corner point1
                        #..........................................
                        #lat1, lon1, t_alt = gps_newpos(lat, lon, 45, 700, ln_param7+c)  #  sq(150)^2xsq(100)^2---->(300x200)- triangle of c side
                        #....lat1, lon1, t_alt = gps_newpos(lat, lon, 45, 300, ln_param7+c)  #  sq(150)^2xsq(100)^2---->(300x200)- triangle of c side
                        #ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4 
                        #0		0		3	16		0		0	0	0
                        UAV6_plat1,UAV6_plon1 = p61lat, p61lon  
                        UAV6_plat2,UAV6_plon2 = p62lat, p62lon   
                        target_bearing1_6 = gps_bearing(UAV6_plat1,UAV6_plon1, UAV6_plat2,UAV6_plon2)  
                        target_distance1_6 = gps_distance(UAV6_plat1,UAV6_plon1, UAV6_plat2,UAV6_plon2)     
                    
                        target_bearing2_6 = gps_bearing(UAV6_plat1,UAV6_plon1, p64lat, p64lon)  
                        target_distance2_6 = gps_distance(UAV6_plat1,UAV6_plon1, p64lat, p64lon)   
                    
                    
                        print ("UAV6_plat1,UAV6_plon1", UAV6_plat1,UAV6_plon1)   
                        print ("UAV6_plat2,UAV6_plon2", UAV6_plat2,UAV6_plon2)  
                        print ("target_bearing1_6", target_bearing1_6)
                        print ("target_distance1_6", target_distance1_6) 
                        print ("target_bearing2_6", target_bearing2_6)
                        print ("target_distance2_6", target_distance2_6)   
       
                        #.........................................search part 1.....................................
                        if number == 1:      
                            print ("....uav1...")
                            #..targetLocation_lat_01, targetLocation_lon_01, targetLocation_alt = gps_newpos(lat1, lon1, 270,0,ln_param7+c) #20,40,60,... uavs spacing, 270 deg uav search placement deg
                            #lat,lon, dist_between_waypoints, heading, num_rows, num_cols  #30
                            UAV1_plat1,UAV1_plon1, targetLocation_alt = gps_newpos(UAV1_plat1, UAV1_plon1, target_bearing2 ,0, ln_param7+c)
                            search_loc(UAV1_plat1,UAV1_plon1, 100, target_bearing1, 6, 1, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4,(ln_param7+c),15)  #(15=300m is search distance, 180 deg search bearing)
                        #.........................................search part 2.....................................
                        elif number == 2:
			     print (".....uav2....")
			     UAV2_plat1,UAV2_plon1, targetLocation_alt = gps_newpos(UAV2_plat1, UAV2_plon1, target_bearing2_2 ,0, ln_param7+c)
			     search_loc(UAV2_plat1,UAV2_plon1, 100, target_bearing1_2, 6, 1, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4,(ln_param7+c),15)  #12.9452939847, 80.1360917556
             			           
                        #.........................................search part 3.....................................
                        elif number == 3:      
                            print ("....uav3...")
                            #..targetLocation_lat_01, targetLocation_lon_01, targetLocation_alt = gps_newpos(lat1, lon1, 270,0,ln_param7+c) #20,40,60,... uavs spacing, 270 deg uav search placement deg
                            UAV3_plat1,UAV3_plon1, targetLocation_alt = gps_newpos(UAV3_plat1, UAV3_plon1, target_bearing2_3 ,0, ln_param7+c)
                            #lat,lon, dist_between_waypoints, heading, num_rows, num_cols
                            search_loc(UAV3_plat1,UAV3_plon1, 100, target_bearing1_3, 6, 1, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4,(ln_param7+c),15)  #(15=300m is search distance, 180 deg search bearing)
                        #.........................................search part 4.....................................
                        elif number == 4:
			     print (".....uav4....")
			     UAV4_plat1,UAV4_plon1, targetLocation_alt = gps_newpos(UAV4_plat1, UAV4_plon1, target_bearing2_4 ,0, ln_param7+c)
			     search_loc(UAV4_plat1,UAV4_plon1, 100, target_bearing1_4, 6, 1, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4,(ln_param7+c),15)  #12.9452939847, 80.1360917556
			     
                        #.........................................search part 4.....................................
                        elif number == 5:
			     print (".....uav5....")
			     UAV5_plat1,UAV5_plon1, targetLocation_alt = gps_newpos(UAV5_plat1, UAV5_plon1, target_bearing2_5 ,0, ln_param7+c)
			     search_loc(UAV5_plat1,UAV5_plon1, 100, target_bearing1_5, 6, 1, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4,(ln_param7+c),15)  #12.9452939847, 80.1360917556
			     
                        #.........................................search part 4.....................................
                        elif number == 6:
			     print (".....uav6....")
			     UAV6_plat1,UAV6_plon1, targetLocation_alt = gps_newpos(UAV6_plat1, UAV6_plon1, target_bearing2_6 ,0, ln_param7+c)
			     search_loc(UAV6_plat1,UAV6_plon1, 100, target_bearing1_6, 6, 1, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4,(ln_param7+c),15)  #12.9452939847, 80.1360917556
			     
                        """
                        elif number == 8:

                            print (".....uav8...")
                            UAV8_plat1,UAV8_plon1, targetLocation_alt = gps_newpos(UAV6_plat1,UAV6_plon1, target_bearing2_2,30,ln_param7+c)
                            search_loc(UAV8_plat1,UAV8_plon1, 100, target_bearing1_2, 10, 1, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4,(ln_param7+c),10) #12.9452939184, 80.1331421644

                        elif number == 9:
			     print ("........uav9......")
			     UAV9_plat1,UAV9_plon1, targetLocation_alt = gps_newpos(UAV6_plat1,UAV6_plon1, target_bearing2_2,40,ln_param7+c)
			     #lat,lon, dist_between_waypoints, heading, num_rows, num_cols
			     search_loc(UAV9_plat1,UAV9_plon1, 100, target_bearing1_2, 10, 1, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4,(ln_param7+c),10)

                        elif number == 10:
			     print ("........uav10.....")
			     UAV10_plat1,UAV10_plon1, targetLocation_alt = gps_newpos(UAV6_plat1,UAV6_plon1, target_bearing2_2, 50, ln_param7+c)
			     search_loc(UAV10_plat1,UAV10_plon1, 100, target_bearing1_2, 10, 1, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4,(ln_param7+c),10)  #12.9452939847, 80.1360917556

                        """

                    elif ln_param1 == 2:
                        print ("payload release pre-defined point")
                        print ("model", model)
                        (lat, lon) = (27.035536, 71.730935)  #center_lat,lon
                        lat1, lon1, t_alt = gps_newpos(lat, lon, 45, 140, ln_param7+c)

                        #lat2, lon2 = gps_newpos(lat, lon, 135, 140)
                        #lat3, lon3 = gps_newpos(lat, lon, 225, 140)
                        #lat4, lon4 = gps_newpos(lat, lon, 315, 140)

                        row_lat1, row_lon1, t_alt = gps_newpos(lat1, lon1, 270, 20, ln_param7+c)
                        row_lat2, row_lon2, t_alt = gps_newpos(lat1, lon1, 270, 40, ln_param7+c)
                        row_lat3, row_lon3, t_alt = gps_newpos(lat1, lon1, 270, 60, ln_param7+c)
                        row_lat4, row_lon4, t_alt = gps_newpos(lat1, lon1, 270, 80, ln_param7+c)
                        row_lat5, row_lon5, t_alt = gps_newpos(lat1, lon1, 270, 100, ln_param7+c)



                        #*****************even***************************

                        print ("sssssssssssssssssss", number)
                        
                        if number == 1:
                            targetLocation_lat, targetLocation_lon, targetLocation_alt = gps_newpos(row_lat1, row_lon1, 180,40,ln_param7+c)
                        
                        elif number == 2:
                            targetLocation_lat, targetLocation_lon, targetLocation_alt = gps_newpos(row_lat1, row_lon1, 180,80,ln_param7+c)
                        elif number == 3:
                            targetLocation_lat, targetLocation_lon, targetLocation_alt = gps_newpos(row_lat2, row_lon2, 180,20,ln_param7+c)
                        elif number == 4:
                            targetLocation_lat, targetLocation_lon, targetLocation_alt = gps_newpos(row_lat2, row_lon2, 180,60,ln_param7+c)
                        elif number == 5:
                            targetLocation_lat, targetLocation_lon, targetLocation_alt = gps_newpos(row_lat2, row_lon2, 180,100,ln_param7+c)
                        elif number == 6:
                            targetLocation_lat, targetLocation_lon, targetLocation_alt = gps_newpos(row_lat3, row_lon3, 180,20,ln_param7+c)
                        elif number == 7:
                            targetLocation_lat, targetLocation_lon, targetLocation_alt = gps_newpos(row_lat3, row_lon3, 180,60,ln_param7+c)
                        elif number == 8:
                            targetLocation_lat, targetLocation_lon, targetLocation_alt = gps_newpos(row_lat3, row_lon3, 180,100,ln_param7+c)
                        elif number == 9:
                            targetLocation_lat, targetLocation_lon, targetLocation_alt = gps_newpos(row_lat4, row_lon4, 180,40,ln_param7+c)
                        elif number == 10:
                            targetLocation_lat, targetLocation_lon, targetLocation_alt = gps_newpos(row_lat4, row_lon4, 180,80,ln_param7+c)
                        
                        elif number == 11:
                            targetLocation_lat, targetLocation_lon, targetLocation_alt = gps_newpos(row_lat1, row_lon1, 180,20,ln_param7+c)
                        elif number == 12:
                            targetLocation_lat, targetLocation_lon, targetLocation_alt = gps_newpos(row_lat1, row_lon1, 180,60,ln_param7+c)
                        elif number == 13:
                            targetLocation_lat, targetLocation_lon, targetLocation_alt = gps_newpos(row_lat1, row_lon1, 180,100,ln_param7+c)
                        elif number == 14:
                            targetLocation_lat, targetLocation_lon, targetLocation_alt = gps_newpos(row_lat2, row_lon2, 180,40,ln_param7+c)
                        elif number == 15:
                            targetLocation_lat, targetLocation_lon, targetLocation_alt = gps_newpos(row_lat2, row_lon2, 180,80,ln_param7+c)
                        elif number == 16:
                            targetLocation_lat, targetLocation_lon, targetLocation_alt = gps_newpos(row_lat3, row_lon3, 180,40,ln_param7+c)
                        elif number == 17:
                            targetLocation_lat, targetLocation_lon, targetLocation_alt = gps_newpos(row_lat3, row_lon3, 180,80,ln_param7+c)
                        elif number == 18:
                            targetLocation_lat, targetLocation_lon, targetLocation_alt = gps_newpos(row_lat4, row_lon4, 180,20,ln_param7+c)
                        elif number == 19:
                            targetLocation_lat, targetLocation_lon, targetLocation_alt = gps_newpos(row_lat4, row_lon4, 180,60,ln_param7+c)
                        elif number == 20:
                            targetLocation_lat, targetLocation_lon, targetLocation_alt = gps_newpos(row_lat4, row_lon4, 180,100,ln_param7+c)
                        #elif number == 21:
                            #targetLocation_lat, targetLocation_lon, targetLocation_alt = gps_newpos(row_lat5, row_lon5, 180,20,ln_param7+c)
                        #elif number == 22:
                            #targetLocation_lat, targetLocation_lon, targetLocation_alt = gps_newpos(row_lat5, row_lon5, 180,40,ln_param7+c)
                        #elif number == 23:
                            #targetLocation_lat, targetLocation_lon, targetLocation_alt = gps_newpos(row_lat5, row_lon5, 180,60,ln_param7+c)
                        """
                        elif number == 24:
                            targetLocation_lat, targetLocation_lon, targetLocation_alt = gps_newpos(row_lat5, row_lon5, 180,80,ln_param7+c)
                        elif number == 25:
                            targetLocation_lat, targetLocation_lon, targetLocation_alt = gps_newpos(row_lat5, row_lon5, 180,100,ln_param7+c)
                        """

                        cmd = Command( 0, 0, 0, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4, targetLocation_lat, targetLocation_lon, targetLocation_alt)

                        missionlist.append(cmd)
			if number == 1:
				missionlist_uav1.append(cmd)
			elif number == 2:
				missionlist_uav2.append(cmd)
			elif number == 3:
				missionlist_uav3.append(cmd)
			elif number == 4:
				missionlist_uav4.append(cmd)
			elif number == 5:
				missionlist_uav5.append(cmd)
			elif number == 6:
				missionlist_uav6.append(cmd)
			elif number == 7:
				missionlist_uav7.append(cmd)
			elif number == 8:
				missionlist_uav8.append(cmd)
			elif number == 9:
				missionlist_uav9.append(cmd)
			elif number == 10:
				missionlist_uav10.append(cmd)
			elif number == 11:
				missionlist_uav11.append(cmd)
			elif number == 12:
				missionlist_uav12.append(cmd)
			elif number == 13:
				missionlist_uav13.append(cmd)
			elif number == 14:
				missionlist_uav14.append(cmd)
			elif number == 15:
				missionlist_uav15.append(cmd)
			elif number == 16:
				missionlist_uav16.append(cmd)
			elif number == 17:
				missionlist_uav17.append(cmd)
			elif number == 18:
				missionlist_uav18.append(cmd)
			elif number == 19:
				missionlist_uav19.append(cmd)
			elif number == 20:
				missionlist_uav20.append(cmd)
			elif number == 21:
				missionlist_uav21.append(cmd)
			elif number == 22:
				missionlist_uav22.append(cmd)
			elif number == 23:
				missionlist_uav23.append(cmd)
			elif number == 24:
				missionlist_uav24.append(cmd)
			elif number == 25:
				missionlist_uav25.append(cmd)
            
                    else:
                        original_location = [ln_param5, ln_param6]
                        if checkboxvalue1.get() == 1:
                            formation(int(no_uavs), 'T', ln_param5, ln_param6)
                        elif checkboxvalue2.get() == 1:
                            formation(int(no_uavs), 'L', ln_param5, ln_param6)
                        elif checkboxvalue3.get() == 1:
                            formation(int(no_uavs), 'S', ln_param5, ln_param6)

                        #init_pos1 = altered_position(original_location,0,0)
                        print ("......latlon_pos", latlon_pos)
                        print (".....len..", len(latlon_pos))
                        print ("....number", number-1)
                        test = latlon_pos[number-1]
                        ##init_pos1 = altered_position(original_location,a,b)
                        ln_param7=float(linearray[10])+c


                        ln_param5 = test[0]
                        ln_param6 = test[1]      
                        ln_autocontinue=int(linearray[11].strip())         
                        cmd = Command( 0, 0, 0, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4, ln_param5, ln_param6, ln_param7)
                        missionlist.append(cmd)
			if number == 1:
				missionlist_uav1.append(cmd)
			elif number == 2:
				missionlist_uav2.append(cmd)
			elif number == 3:
				missionlist_uav3.append(cmd)
			elif number == 4:
				missionlist_uav4.append(cmd)
			elif number == 5:
				missionlist_uav5.append(cmd)
			elif number == 6:
				missionlist_uav6.append(cmd)
			elif number == 7:
				missionlist_uav7.append(cmd)
			elif number == 8:
				missionlist_uav8.append(cmd)
			elif number == 9:
				missionlist_uav9.append(cmd)
			elif number == 10:
				missionlist_uav10.append(cmd)
			elif number == 11:
				missionlist_uav11.append(cmd)
			elif number == 12:
				missionlist_uav12.append(cmd)
			elif number == 13:
				missionlist_uav13.append(cmd)
			elif number == 14:
				missionlist_uav14.append(cmd)
			elif number == 15:
				missionlist_uav15.append(cmd)
			elif number == 16:
				missionlist_uav16.append(cmd)
			elif number == 17:
				missionlist_uav17.append(cmd)
			elif number == 18:
				missionlist_uav18.append(cmd)
			elif number == 19:
				missionlist_uav19.append(cmd)
			elif number == 20:
				missionlist_uav20.append(cmd)
			elif number == 21:
				missionlist_uav21.append(cmd)
			elif number == 22:
				missionlist_uav22.append(cmd)
			elif number == 23:
				missionlist_uav23.append(cmd)
			elif number == 24:
				missionlist_uav24.append(cmd)
			elif number == 25:
				missionlist_uav25.append(cmd)
    if number == 1:
    	missionlist_uav_all.append(missionlist_uav1)
    elif number == 2:
    	missionlist_uav_all.append(missionlist_uav2)
    elif number == 3:
    	missionlist_uav_all.append(missionlist_uav3)
    elif number == 4:
    	missionlist_uav_all.append(missionlist_uav4)
    elif number == 5:
    	missionlist_uav_all.append(missionlist_uav5)
    elif number == 6:
    	missionlist_uav_all.append(missionlist_uav6)
    elif number == 7:
    	missionlist_uav_all.append(missionlist_uav7)
    elif number == 8:
    	missionlist_uav_all.append(missionlist_uav8)
    elif number == 9:
    	missionlist_uav_all.append(missionlist_uav9)
    elif number == 10:
    	missionlist_uav_all.append(missionlist_uav10)
    elif number == 11:
    	missionlist_uav_all.append(missionlist_uav11)
    elif number == 12:
    	missionlist_uav_all.append(missionlist_uav12)
    elif number == 13:
    	missionlist_uav_all.append(missionlist_uav13)
    elif number == 14:
    	missionlist_uav_all.append(missionlist_uav14)
    elif number == 15:
    	missionlist_uav_all.append(missionlist_uav15)
    elif number == 16:
    	missionlist_uav_all.append(missionlist_uav16)
    elif number == 17:
    	missionlist_uav_all.append(missionlist_uav17)
    elif number == 18:
    	missionlist_uav_all.append(missionlist_uav18)
    elif number == 19:
    	missionlist_uav_all.append(missionlist_uav19)
    elif number == 20:
    	missionlist_uav_all.append(missionlist_uav20)
    elif number == 21:
    	missionlist_uav_all.append(missionlist_uav21)
    elif number == 22:
    	missionlist_uav_all.append(missionlist_uav22)
    elif number == 23:
    	missionlist_uav_all.append(missionlist_uav23)
    elif number == 24:
    	missionlist_uav_all.append(missionlist_uav24)
    elif number == 25:
    	missionlist_uav_all.append(missionlist_uav25)

    return missionlist


def cmdline(vehicle,missionlist):
  if vehicle == None:
        print ("slave is lost")
  else:
    print("\nUpload mission from a file: %s" % export_mission_filename)
    print(' Clear mission')
    cmds = vehicle.commands
    cmds.clear()   

    #Add new mission to vehicle

    for command in missionlist:
        cmds.add(command)
   
    print('Upload mission to master')
    vehicle.commands.upload()  

def download_mission():
    global master
    if master == 1:
	    print(" Download mission from vehicle")
	    missionlist=[]
	    cmds = vehicle1.commands
	    cmds.download()
	    cmds.wait_ready()
	    for cmd in cmds:
		missionlist.append(cmd)
	    return missionlist
    if master == 2:
	    print(" Download mission from vehicle")
	    missionlist=[]
	    cmds = vehicle2.commands
	    cmds.download()
	    cmds.wait_ready()
	    for cmd in cmds:
		missionlist.append(cmd)
	    return missionlist
    if master == 3:
	    print(" Download mission from vehicle")
	    missionlist=[]
	    cmds = vehicle3.commands
	    cmds.download()
	    cmds.wait_ready()
	    for cmd in cmds:
		missionlist.append(cmd)
	    return missionlist	 
    if master == 4:
	    print(" Download mission from vehicle")
	    missionlist=[]
	    cmds = vehicle4.commands
	    cmds.download()
	    cmds.wait_ready()
	    for cmd in cmds:
		missionlist.append(cmd)
	    return missionlist
    if master == 5:
	    print(" Download mission from vehicle")
	    missionlist=[]
	    cmds = vehicle5.commands
	    cmds.download()
	    cmds.wait_ready()
	    for cmd in cmds:
		missionlist.append(cmd)
	    return missionlist
    if master == 6:
	    print(" Download mission from vehicle")
	    missionlist=[]
	    cmds = vehicle6.commands
	    cmds.download()
	    cmds.wait_ready()
	    for cmd in cmds:
		missionlist.append(cmd)
	    return missionlist

def send_mission_to_uav_all():
    global slave_heal_ip
    for i in range(0, len(slave_heal_ip)):
	    if slave_heal_ip[i] == 'nolink':
		print ("nolink")
	    else:
		    print ("....slave_heal_ip..", slave_heal_ip[i])
		    for m in range(0, 3): 
			# Now we can create socket object
			s = socket.socket()
			# Lets choose one port and connect to that port
			PORT = 9898
			# Lets connect to that port where server may be running
			s.connect((slave_heal_ip[i], PORT))
			# We can send file sample.txt
		    	aFileName = 'mission_uav'+str(m+1)+'.'+'txt'
			print ("...aFileName...", aFileName)
			file = open(aFileName, "rb")
			SendData = file.read(4096)
			while SendData:
			    # Now we can receive data from server
			    #print("\n\n################## Below message is received from server ################## \n\n ", s.recv(1024).decode("utf-8"))
			    print (s.recv(4096))
			    #Now send the content of sample.txt to server
			    s.send(SendData)
			    SendData = file.read(4096)      

			# Close the connection from client side
			s.close()

def save_mission_to_uav_all():
    global follower_host_tuple
    global missionlist_uav_all

    print ("missionlist_uav_all", missionlist_uav_all)
    #output='QGC WPL 110\n'
    #for i, iter_follower_mission in enumerate(missionlist_uav_all): 
    for i in range (0, no_uavs): 
    	aFileName = 'mission_uav'+str(i+1)+'.'+'txt'
	iter_follower_mission = None
	print (">>>>>>>>>>>>>>>>>>>>>...........", i)
	print ("missionlist_uav_all", missionlist_uav_all[i])
	iter_follower_mission = missionlist_uav_all[i]
	output = None
	output='QGC WPL 110\n'
    	for cmd in iter_follower_mission:
		commandline="%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (cmd.seq,cmd.current,cmd.frame,cmd.command,cmd.param1,cmd.param2,cmd.param3,cmd.param4,cmd.x,cmd.y,cmd.z,cmd.autocontinue)
		print ("commandline", commandline)
		output+=commandline
		
    	with open(aFileName, 'w') as file_:
		print(" Write mission to file", aFileName)
		file_.write(output)


"""
def save_mission(aFileName):
    global vehicle1
    print("\nSave mission from Vehicle to file: %s" % export_mission_filename)    
    #Download mission from vehicle
    missionlist = download_mission()
    #Add file-format information
    output='QGC WPL 110\n'
    #Add home location as 0th waypoint
    home = vehicle1.home_location
    output+="%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (0,1,0,16,0,0,0,0,home.lat,home.lon,home.alt,1)
    #Add commands
    for cmd in missionlist:
        commandline="%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (cmd.seq,cmd.current,cmd.frame,cmd.command,cmd.param1,cmd.param2,cmd.param3,cmd.param4,cmd.x,cmd.y,cmd.z,cmd.autocontinue)
        output+=commandline
    with open(aFileName, 'w') as file_:
        print(" Write mission to file")
        file_.write(output)
"""

def download_mission_1():
    global master
    global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10
    if master == 1:
	    print(" Download mission from vehicle")
	    missionlist=[]
	    cmds = vehicle1.commands
	    cmds.download()
	    cmds.wait_ready()
	    for cmd in cmds:
		missionlist.append(cmd)
	    return missionlist
    if master == 2:
	    print(" Download mission from vehicle")
	    missionlist=[]
	    cmds = vehicle2.commands
	    cmds.download()
	    cmds.wait_ready()
	    for cmd in cmds:
		missionlist.append(cmd)
	    return missionlist
    if master == 3:
	    print(" Download mission from vehicle")
	    missionlist=[]
	    cmds = vehicle3.commands
	    cmds.download()
	    cmds.wait_ready()
	    for cmd in cmds:
		missionlist.append(cmd)
	    return missionlist	 
    if master == 4:
	    print(" Download mission from vehicle")
	    missionlist=[]
	    cmds = vehicle4.commands
	    cmds.download()
	    cmds.wait_ready()
	    for cmd in cmds:
		missionlist.append(cmd)
	    return missionlist
    if master == 5:
	    print(" Download mission from vehicle")
	    missionlist=[]
	    cmds = vehicle5.commands
	    cmds.download()
	    cmds.wait_ready()
	    for cmd in cmds:
		missionlist.append(cmd)
	    return missionlist
    if master == 6:
	    print(" Download mission from vehicle")
	    missionlist=[]
	    cmds = vehicle6.commands
	    cmds.download()
	    cmds.wait_ready()
	    for cmd in cmds:
		missionlist.append(cmd)
	    return missionlist

def download_mission_Guided():
    global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10, master


    aFileName = 'exportedmission_01.txt'
    #Download mission from vehicle

    #Add file-format information
    output='QGC WPL 110\n'
    #Add home location as 0th waypoint
    if master == 1:
	    lat = vehicle1.location.global_relative_frame.lat
	    lon = vehicle1.location.global_relative_frame.lon
	    alt_0 = vehicle1.location.global_relative_frame.alt
	    missionlist = download_mission_1()
	    home = vehicle1.home_location

    if master == 2:
	    lat = vehicle2.location.global_relative_frame.lat
	    lon = vehicle2.location.global_relative_frame.lon
	    alt_0 = vehicle2.location.global_relative_frame.alt
	    missionlist = download_mission_1()
	    home = vehicle2.home_location

    if master == 3:
	    lat = vehicle3.location.global_relative_frame.lat
	    lon = vehicle3.location.global_relative_frame.lon
	    alt_0 = vehicle3.location.global_relative_frame.alt
	    missionlist = download_mission_1()
	    home = vehicle3.home_location
		 
    if master == 4:
	    lat = vehicle4.location.global_relative_frame.lat
	    lon = vehicle4.location.global_relative_frame.lon
	    alt_0 = vehicle4.location.global_relative_frame.alt
	    missionlist = download_mission_1()
	    home = vehicle4.home_location

    if master == 5:
	    lat = vehicle5.location.global_relative_frame.lat
	    lon = vehicle5.location.global_relative_frame.lon
	    alt_0 = vehicle5.location.global_relative_frame.alt
	    missionlist = download_mission_1()
	    home = vehicle5.home_location

    if master == 6:
	    lat = vehicle6.location.global_relative_frame.lat
	    lon = vehicle6.location.global_relative_frame.lon
	    alt_0 = vehicle6.location.global_relative_frame.alt
	    missionlist = download_mission_1()
	    home = vehicle6.home_location

    output+="%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (0,1,0,16,0,0,0,0,home.lat,home.lon,home.alt,1)
    #Add commands
    for cmd in missionlist:
        commandline="%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (cmd.seq,cmd.current,cmd.frame,cmd.command,cmd.param1,cmd.param2,cmd.param3,cmd.param4,cmd.x,cmd.y,cmd.z,cmd.autocontinue)
        output+=commandline
	print ("commandline",commandline)
    with open(aFileName, 'w') as file_:
        print(" exportedmission_01.txt Write mission to file")
        file_.write(output)


def generate_search_misison_1():
	global master
	export_mission_filename = 'exportedmission.txt'
	if master == 1:		
    		save_mission(vehicle1, export_mission_filename)
	elif master == 2:		
    		save_mission(vehicle2, export_mission_filename)
	elif master == 2:		
    		save_mission(vehicle3, export_mission_filename)

	time.sleep(1)
	 
	aFileName = export_mission_filename

    	with open(aFileName) as f:
		for i, line in enumerate(f):  
		    if i==0:
		        if not line.startswith('QGC WPL 110'):
		            raise Exception('File is not supported WP version')
		    elif i==1:
		            print ("first way point reject")
		    else:
                    
		            linearray=line.split('\t')
		            ln_index=int(linearray[0])
		            ln_currentwp=int(linearray[1])
		            ln_frame=int(linearray[2])
		            ln_command=int(linearray[3])
		            ln_param1=float(linearray[4])
		            ln_param2=float(linearray[5])
		            ln_param3=float(linearray[6])
		            ln_param4=float(linearray[7])
		            ln_param5=float(linearray[8])
		            ln_param6=float(linearray[9])
		            ln_param7=float(linearray[10])
		            ln_autocontinue=int(linearray[11].strip())
		            original_location = [ln_param5, ln_param6]
		            print ("ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4", ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4)
			


def generate_search_misison():
#def waypoint():
	global self_heal, follower_host_tuple
        global search_no_time
        salt = salt_entry.get() 
        salt = int(salt)


	global missionlist, home_locations, no_uavs
	global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10, vehicle11, vehicle12, vehicle13, vehicle14,vehicle15,vehicle16,vehicle17,vehicle18,vehicle19,vehicle20,vehicle21,vehicle22,vehicle23,vehicle24,vehicle25

	ln_param7 = 50
	c = 0

	missionlist_8_all = []
	missionlist_7_all = []
	missionlist_6_all = []
	missionlist_5_all = []
	missionlist_4_all = []
	missionlist_3_all = []
	missionlist_2_all = []
	missionlist_1_all = []

	wayPoints_polygon_1 = []
	wayPoints_polygon_2 = []
	wayPoints_polygon_3 = []
	wayPoints_polygon_4 = []
	wayPoints_polygon_5 = []
	wayPoints_polygon_6 = []
	wayPoints_polygon_7 = []
	wayPoints_polygon_8 = []
	wayPoints_polygon_78 = []

	wayPoints_polygon_pos_1 = []
	wayPoints_polygon_pos_2 = []
	wayPoints_polygon_pos_3 = []
	wayPoints_polygon_pos_4 = []
	wayPoints_polygon_pos_5 = []
	wayPoints_polygon_pos_6 = []
	wayPoints_polygon_pos_7 = []
	wayPoints_polygon_pos_8 = []
	wayPoints_polygon_pos_78 = []

	wayPoints_lat_lon_uav1 = []
	wayPoints_lat_lon_uav2 = []
	wayPoints_lat_lon_uav3 = []
	wayPoints_lat_lon_uav4 = []
	wayPoints_lat_lon_uav5 = []
	wayPoints_lat_lon_uav6 = []
	wayPoints_lat_lon_uav7 = []
	wayPoints_lat_lon_uav8 = []
	wayPoints_lat_lon_uav78 = []

	missionlist_1 = []
	missionlist_2 = []
	missionlist_3 = []
	missionlist_4 = []
	missionlist_5 = []
	missionlist_6 = []
	missionlist_7 = []
	missionlist_8 = []
	missionlist_78 = []

	#save_mission(export_mission_filename)

	print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Target search area")
	if circle_pos_flag == True:
                clat = clat_entry.get()
                clon = clon_entry.get()
                clat = float(clat)
                clon = float(clon)

	#(lat, lon) = (12.894874, 80.0544012)  #center_lat,lon   
	(lat, lon) = (clat, clon)  #center_lat,lon   
        try:
		grid_level = search_aera_set_entry.get() 
		grid_level = int(grid_level)
	except:
		grid_level = 12

	try:
		map_zoomLevel = scaleset_entry.get() 
		map_zoomLevel = int(map_zoomLevel)
	except:
 		map_zoomLevel = 17
	print ("...self_heal..", self_heal)
        
	count_swarm = 0
        
	for i in range(0, len(self_heal)):
		if (int(self_heal[i]) > 0):
			print ("lost uav for swarm")
		else:
			count_swarm = count_swarm+1
	print ("....no of uav avilable for swarm", count_swarm)
        
	try:
		angle_grid = search_count_set_entry.get() 
		angle_grid = int(angle_grid)
	except:

		angle_grid = 17
        

        scale = 1
	"""
	angle_grid = 0
	map_zoomLevel = 17
	grid_line_space = 0
	"""
	if 0 <= grid_level <= 2:
		print ("grid space 17")
		grid_line_space = 20
	if 2 < grid_level <= 5:
		print ("grid space 17")
		grid_line_space = 19
	if 5 < grid_level <= 10:
		print ("grid space 17")
		grid_line_space = 18
	if 10 < grid_level <= 15:
		print ("grid space 17")
		grid_line_space = 17
	if 15 < grid_level <= 20:
		print ("grid space 16")
		grid_line_space = 16
	
	center_lat, center_lon = (lat, lon)  #center_lat,lo
	WIDTH_C, HEIGHT_C = 1920, 1080
	if count_swarm == 8:
		#########..part 1.........
		#......................square 1(250x500)...................
		ext_1 = []
		survey_agri_grid_lat_lon = [(12.997399,80.181902), (12.997903,80.182291), (12.997812,80.182447), (12.997313,80.182026)]
		point_func = 'AB_point'
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)
			point_ll1 = (latitude, longitude)
			wayPoints_lat_lon_uav1.append(point_ll1)

			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude
                        if search_no_time == 1:                      

				print ("....8888......iiiiiiiiii", i)
				if i == 0:
					ln_param7 = salt
				elif i == 1:
					ln_param7 = salt

				elif i == len(x):
					ln_param7 = salt
				elif i == len(x)-1:
					ln_param7 = salt

				else:
					if search_no_time == 1:   #same alt search
						ln_param7 = 100
					else:
						ln_param7 = salt
			
				
			#cmd = Command( 0, 0, 0, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4, ln_param5, ln_param6, ln_param7)
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_1.append(cmd)
		print ("....uav1...grid_wp..", wayPoints_lat_lon_uav1)
		print ("....leng...", len(wayPoints_lat_lon_uav1))
		#....cmdline(vehicle1,missionlist_1)

		missionlist_8_all.append(missionlist_1)
		
	#......................square 2(250x500)...................
		ext_1 = []		
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.9979,80.1822), (12.9974,80.1819), (12.9973,80.1820), (12.9978,80.1824)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)
			point_ll2 = (latitude, longitude)
			wayPoints_lat_lon_uav2.append(point_ll2)

			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude

                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+10
				elif i == 1:
					ln_param7 = salt+10

				elif i == len(x):
					ln_param7 = salt+10
				elif i == len(x)-1:
					ln_param7 = salt+10
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100
					else:
						ln_param7 = salt+10
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_2.append(cmd)
		print ("....uav2...grid_wp..", wayPoints_lat_lon_uav2)

		missionlist_8_all.append(missionlist_2)

		#......................square 3(250x500)...................
		ext_1 = []
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.9979,80.1822), (12.9974,80.1819), (12.9973,80.1820), (12.9978,80.1824)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)
			point_ll3 = (latitude, longitude)
			wayPoints_lat_lon_uav3.append(point_ll3)

			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude

                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+20
				elif i == 1:
					ln_param7 = salt+20

				elif i == len(x):
					ln_param7 = salt+20
				elif i == len(x)-1:
					ln_param7 = salt+20

				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100
					else:
						ln_param7 = salt+20
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_3.append(cmd)
		print ("....uav3...grid_wp..", wayPoints_lat_lon_uav3)

		missionlist_8_all.append(missionlist_3)

		#......................square 4(250x500)...................
		ext_1 = []
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.9979,80.1822), (12.9974,80.1819), (12.9973,80.1820), (12.9978,80.1824)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)

			point_ll4 = (latitude, longitude)
			wayPoints_lat_lon_uav1.append(point_ll4)

			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude

                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+30
				elif i == 1:
					ln_param7 = salt+30

				elif i == len(x):
					ln_param7 = salt+30
				elif i == len(x)-1:
					ln_param7 = salt+30

				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100
					else:
						ln_param7 = salt+30
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_4.append(cmd)
		print ("....uav4...grid_wp..", wayPoints_lat_lon_uav4)

		missionlist_8_all.append(missionlist_4)


		#########..part 2.........

		#......................square 5(250x500)...................
		ext_1 = []
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.9979,80.1822), (12.9974,80.1819), (12.9973,80.1820), (12.9978,80.1824)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)
			point_ll5 = (latitude, longitude)
			wayPoints_lat_lon_uav1.append(point_ll5)

			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude

                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+40
				elif i == 1:
					ln_param7 = salt+40

				elif i == len(x):
					ln_param7 = salt+40
				elif i == len(x)-1:
					ln_param7 = salt+40

				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100
					else:
						ln_param7 = salt+40
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_5.append(cmd)
		print ("....uav5...grid_wp..", wayPoints_lat_lon_uav5)

		missionlist_8_all.append(missionlist_5)

		#......................square 6(250x500)...................
		ext_1 = []
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.9979,80.1822), (12.9974,80.1819), (12.9973,80.1820), (12.9978,80.1824)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)

			point_ll6 = (latitude, longitude)
			wayPoints_lat_lon_uav1.append(point_ll6)
			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude

                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+50
				elif i == 1:
					ln_param7 = salt+50

				elif i == len(x):
					ln_param7 = salt+50
				elif i == len(x)-1:
					ln_param7 = salt+50

				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100
					else:
						ln_param7 = salt+50
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_6.append(cmd)
		print ("....uav6...grid_wp..", wayPoints_lat_lon_uav6)

		missionlist_8_all.append(missionlist_6)

		#......................square 7(250x500)...................
		ext_1 = []
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.9979,80.1822), (12.9974,80.1819), (12.9973,80.1820), (12.9978,80.1824)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)
			point_ll7 = (latitude, longitude)
			wayPoints_lat_lon_uav1.append(point_ll7)

			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude

                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+60
				if i == 1:
					ln_param7 = salt+60
				elif i == len(x):
					ln_param7 = salt+60
				elif i == len(x)-1:
					ln_param7 = salt+60
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100
					else:
						ln_param7 = salt+60
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_7.append(cmd)
		print ("....uav7...grid_wp..", wayPoints_lat_lon_uav7)

		missionlist_8_all.append(missionlist_7)


		#......................square 8(250x500)...................
		ext_1 = []
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.9979,80.1822), (12.9974,80.1819), (12.9973,80.1820), (12.9978,80.1824)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)
			point_ll8 = (latitude, longitude)
			wayPoints_lat_lon_uav1.append(point_ll8)

			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude

                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+70
				if i == 1:
					ln_param7 = salt+70
				elif i == len(x):
					ln_param7 = salt+70
				elif i == len(x)-1:
					ln_param7 = salt+70
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100
					else:
						ln_param7 = salt+70
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_8.append(cmd)

		missionlist_8_all.append(missionlist_8)


		print ("....uav8...grid_wp..", wayPoints_lat_lon_uav8)
		print ("....leng...", len(wayPoints_lat_lon_uav8))


		#............................
                follower_host_tuple_heal = []
		for j, iter_follower_heal in enumerate(follower_host_tuple): 
			print (".....j.....", j)
			if int(self_heal[j]) > 0:
				print (">>>>>..lost uav for swarm")
			else:
				print ("....ok....")
				follower_host_tuple_heal.append(iter_follower_heal)

		#........................
		print ("...follower_host_tuple_heal....", follower_host_tuple_heal)
		for j, iter_follower in enumerate(follower_host_tuple_heal): 
			print (".....jjj...", j)
			cmdline(iter_follower,missionlist_8_all[j])
			print ("...%%%%....")
		#.......................

	
	if count_swarm == 7:
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude

                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt
				if i == 1:
					ln_param7 = salt
				elif i == len(x):
					ln_param7 = salt
				elif i == len(x)-1:
					ln_param7 = salt
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100

					else:
						ln_param7 = salt

			else:
				if i == 0:
					ln_param7 = salt
				elif i == len(x):
					ln_param7 = salt
				elif i == len(x)-1:
					ln_param7 = salt
				else:
					ln_param7 = 100
				
			#cmd = Command( 0, 0, 0, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4, ln_param5, ln_param6, ln_param7)
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_1.append(cmd)
		print ("....uav1...grid_wp..", wayPoints_lat_lon_uav1)
		print ("....leng...", len(wayPoints_lat_lon_uav1))
		missionlist_7_all.append(missionlist_1)
		
	#......................square 2(250x600)...................
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude

                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+10
				if i == 1:
					ln_param7 = salt+10
				elif i == len(x):
					ln_param7 = (salt+10)
				elif i == len(x)-1:
					ln_param7 = salt+10

				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100
					else:
						ln_param7 = salt+10

			else:
				if i == 0:
					ln_param7 = salt+10
				elif i == len(x):
					ln_param7 = salt+10
				elif i == len(x)-1:
					ln_param7 = salt+10
				else:
					ln_param7 = 100

			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_2.append(cmd)
		print ("....uav2...grid_wp..", wayPoints_lat_lon_uav2)
		missionlist_7_all.append(missionlist_2)

		#......................square 3(250x600)...................
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude
                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+20
				if i == 1:
					ln_param7 = salt+20
				elif i == len(x):
					ln_param7 = salt+20
				elif i == len(x)-1:
					ln_param7 = salt+20
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100
					else:
						ln_param7 = salt+20

			else:
				if i == 0:
					ln_param7 = salt+20
				elif i == len(x):
					ln_param7 = salt+20
				elif i == len(x)-1:
					ln_param7 = salt+20
				else:
					ln_param7 = 100
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_3.append(cmd)
		print ("....uav3...grid_wp..", wayPoints_lat_lon_uav3)
		missionlist_7_all.append(missionlist_3)

		#......................square 4(250x600)...................
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude
                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+30
				if i == 1:
					ln_param7 = salt+30
				elif i == len(x):
					ln_param7 = salt+30
				elif i == len(x)-1:
					ln_param7 = salt+30
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100
					else:
						ln_param7 = salt+30

			else:
				if i == 0:
					ln_param7 = salt+30
				elif i == len(x):
					ln_param7 = salt+30
				elif i == len(x)-1:
					ln_param7 = salt+30
				else:
					ln_param7 = 100
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_4.append(cmd)
		print ("....uav4...grid_wp..", wayPoints_lat_lon_uav4)
		missionlist_7_all.append(missionlist_4)


		#########..part 2.........

		#......................square 5(333x400)...................
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude
                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+40
				if i == 1:
					ln_param7 = salt+40
				elif i == len(x):
					ln_param7 = salt+40
				elif i == len(x)-1:
					ln_param7 = salt+40
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100
					else:
						ln_param7 = salt+40

			else:
				if i == 0:
					ln_param7 = salt+40
				elif i == len(x):
					ln_param7 = salt+40
				elif i == len(x)-1:
					ln_param7 = salt+40
				else:
					ln_param7 = 100
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_5.append(cmd)
		print ("....uav5...grid_wp..", wayPoints_lat_lon_uav5)
		missionlist_7_all.append(missionlist_5)

		#......................square 6(333x400)...................
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude

                        if search_no_time == 1:                      

				if i == 0:
					ln_param7 = salt+50
				if i == 1:
					ln_param7 = salt+50
				elif i == len(x):
					ln_param7 = salt+50
				elif i == len(x)-1:
					ln_param7 = salt+50
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100
					else:
						ln_param7 = salt+50

			else:
				if i == 0:
					ln_param7 = salt+50
				elif i == len(x):
					ln_param7 = salt+50
				elif i == len(x)-1:
					ln_param7 = salt+50
				else:
					ln_param7 = 100
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_6.append(cmd)
		print ("....uav6...grid_wp..", wayPoints_lat_lon_uav6)
		missionlist_7_all.append(missionlist_6)

		#......................square 7(333x400)...................
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude

                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+60
				if i == 1:
					ln_param7 = salt+60
				elif i == len(x):
					ln_param7 = salt+60
				elif i == len(x)-1:
					ln_param7 = salt+60
				else:
					if search_no_time == 1:   #same alt search
						ln_param7 = 100
					else:
						ln_param7 = salt+60
			else:
				if i == 0:
					ln_param7 = salt+60
				elif i == len(x):
					ln_param7 = salt+60
				elif i == len(x)-1:
					ln_param7 = salt+60
				else:
					ln_param7 = 100
				#ln_param7 = 300
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_7.append(cmd)
		print ("....uav7...grid_wp..", wayPoints_lat_lon_uav7)

		missionlist_7_all.append(missionlist_7)

		#............................
                follower_host_tuple_heal = []
		for j, iter_follower_heal in enumerate(follower_host_tuple): 
			print (".....j.....", j)
			if int(self_heal[j]) > 0:
				print (">>>>>..lost uav for swarm")
			else:
				print ("....ok....")
				follower_host_tuple_heal.append(iter_follower_heal)

		#........................
		print ("...follower_host_tuple_heal....", follower_host_tuple_heal)
		for j, iter_follower in enumerate(follower_host_tuple_heal): 
			print (".....jjj...", j)
			cmdline(iter_follower,missionlist_7_all[j])
			print ("...%%%%....")
		#.......................
		
		print ("....wp uploading is done ..")
		print ("all uav changed to to search")
		

	if count_swarm == 6:
		Tdis_6 = sqrt(pow(search_area,2)+pow(int(search_area/6),2))
		angle_6 = math.degrees(math.atan2(search_area, int(search_area/6)))

		print ("Tdis_6", Tdis_6)
		print ("angle_6",angle_6)
		#......................square 1(333x500)...................
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude
                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt
				if i == 1:
					ln_param7 = salt
				elif i == len(x):
					ln_param7 = salt
				elif i == len(x)-1:
					ln_param7 = salt
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100

					else:
						ln_param7 = salt
			else:
				if i == 0:
					ln_param7 = salt
				elif i == len(x):
					ln_param7 = salt
				elif i == len(x)-1:
					ln_param7 = salt
				else:
					ln_param7 = 100

				#ln_param7 = 300
			#cmd = Command( 0, 0, 0, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4, ln_param5, ln_param6, ln_param7)
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_1.append(cmd)
		print ("....uav1...grid_wp..", wayPoints_lat_lon_uav1)
		print ("....leng...", len(wayPoints_lat_lon_uav1))
		missionlist_6_all.append(missionlist_1)
		
	#......................square 2(333x500)...................
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude
                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+10
				if i == 1:
					ln_param7 = salt+10
				elif i == len(x):
					ln_param7 = salt+10
				elif i == len(x)-1:
					ln_param7 = salt+10
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100

					else:
						ln_param7 = salt+10

			else:
				if i == 0:
					ln_param7 = salt+10
				elif i == len(x):
					ln_param7 = salt+10
				elif i == len(x)-1:
					ln_param7 = salt+10
				else:
					ln_param7 = 100
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_2.append(cmd)
		print ("....uav2...grid_wp..", wayPoints_lat_lon_uav2)
		missionlist_6_all.append(missionlist_2)

		#......................square 3(333x500)...................
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude
                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+20
				if i == 1:
					ln_param7 = salt+20
				elif i == len(x):
					ln_param7 = salt+20
				elif i == len(x)-1:
					ln_param7 = salt+20
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100

					else:
						ln_param7 = salt+20
			else:
				if i == 0:
					ln_param7 = salt+20
				elif i == len(x):
					ln_param7 = salt+20
				elif i == len(x)-1:
					ln_param7 = salt+20
				else:
					ln_param7 = 100
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_3.append(cmd)
		print ("....uav3...grid_wp..", wayPoints_lat_lon_uav3)
		missionlist_6_all.append(missionlist_3)

		#......................square 4(333x500)...................
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude
                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+30
				if i == 1:
					ln_param7 = salt+30
				elif i == len(x):
					ln_param7 = salt+30
				elif i == len(x)-1:
					ln_param7 = salt+30
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100

					else:
						ln_param7 = salt+30

			else:
				if i == 0:
					ln_param7 = salt+30
				elif i == len(x):
					ln_param7 = salt+30
				elif i == len(x)-1:
					ln_param7 = salt+30
				else:
					ln_param7 = 100

			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_4.append(cmd)
		print ("....uav4...grid_wp..", wayPoints_lat_lon_uav4)
		missionlist_6_all.append(missionlist_4)


		#########..part 2.........

		#......................square 5(333x500)...................
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude
                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+40
				if i == 1:
					ln_param7 = salt+40
				elif i == len(x):
					ln_param7 = salt+40
				elif i == len(x)-1:
					ln_param7 = salt+40
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100

					else:
						ln_param7 = salt+40

			else:
				if i == 0:
					ln_param7 = salt+40
				elif i == len(x):
					ln_param7 = salt+40
				elif i == len(x)-1:
					ln_param7 = salt+40
				else:
					ln_param7 = 100

			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_5.append(cmd)
		print ("....uav5...grid_wp..", wayPoints_lat_lon_uav5)
		missionlist_6_all.append(missionlist_5)

		#......................square 6(333x500)...................
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude

                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+50
				if i == 1:
					ln_param7 = salt+50
				elif i == len(x):
					ln_param7 = salt+50
				elif i == len(x)-1:
					ln_param7 = salt+50
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100

					else:
						ln_param7 = salt+50

			else:
				if i == 0:
					ln_param7 = salt+50
				elif i == len(x):
					ln_param7 = salt+50
				elif i == len(x)-1:
					ln_param7 = salt+50
				else:
					ln_param7 = 100

			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_6.append(cmd)
		print ("....uav6...grid_wp..", wayPoints_lat_lon_uav6)
		missionlist_6_all.append(missionlist_6)

		#........................
		#............................
                follower_host_tuple_heal = []
		for j, iter_follower_heal in enumerate(follower_host_tuple): 
			print (".....j.....", j)
			if int(self_heal[j]) > 0:
				print (">>>>>..lost uav for swarm")
			else:
				print ("....ok....")
				follower_host_tuple_heal.append(iter_follower_heal)

		#........................
		print ("...follower_host_tuple_heal....", follower_host_tuple_heal)
		for j, iter_follower in enumerate(follower_host_tuple_heal): 
			print (".....jjj...", j)
			cmdline(iter_follower,missionlist_6_all[j])
			print ("...%%%%....")
		#.......................			
		print ("....wp uploading is done ..")
		print ("all uav changed to to search")
#...............................

	if count_swarm == 5:
		Tdis_5 = sqrt(pow(search_area,2)+pow(int(search_area/5),2))
		angle_5 = math.degrees(math.atan2(search_area, int(search_area/5)))
		print ("Tdis_5", Tdis_5)
		print ("angle_5",angle_5)
		#########..part 1.........
		#......................square 1(500x400)...................
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude

		        print ("....5555......iiiiiiiiii", i)
                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt
				if i == 1:
					ln_param7 = salt
				elif i == len(x):
					ln_param7 = salt
				elif i == len(x)-1:
					ln_param7 = salt
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100

					else:
						ln_param7 = salt

			else:
				if i == 0:
					ln_param7 = salt
				elif i == len(x):
					ln_param7 = salt
				elif i == len(x)-1:
					ln_param7 = salt
				else:
					ln_param7 = 100
				#ln_param7 = 300
			#cmd = Command( 0, 0, 0, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4, ln_param5, ln_param6, ln_param7)
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_1.append(cmd)
		print ("....uav1...grid_wp..", wayPoints_lat_lon_uav1)
		print ("....leng...", len(wayPoints_lat_lon_uav1))
		missionlist_5_all.append(missionlist_1)
		
	#......................square 2(500x400)...................
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude
                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+10
				if i == 1:
					ln_param7 = salt+10
				elif i == len(x):
					ln_param7 = salt+10
				elif i == len(x)-1:
					ln_param7 = salt+10
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100

					else:
						ln_param7 = salt+10

			else:
				if i == 0:
					ln_param7 = salt+10
				elif i == len(x):
					ln_param7 = salt+10
				elif i == len(x)-1:
					ln_param7 = salt+10
				else:
					ln_param7 = 100
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_2.append(cmd)
		print ("....uav2...grid_wp..", wayPoints_lat_lon_uav2)
		missionlist_5_all.append(missionlist_2)

		#......................square 3(333x600)...................
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude
                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+20
				if i == 1:
					ln_param7 = salt+20
				elif i == len(x):
					ln_param7 = salt+20
				elif i == len(x)-1:
					ln_param7 = salt+20
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100

					else:
						ln_param7 = salt+20

			else:
				if i == 0:
					ln_param7 = salt+20
				elif i == len(x):
					ln_param7 = salt+20
				elif i == len(x)-1:
					ln_param7 = salt+20
				else:
					ln_param7 = 100
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_3.append(cmd)
		print ("....uav3...grid_wp..", wayPoints_lat_lon_uav3)
		missionlist_5_all.append(missionlist_3)

		#......................square 4(333x600)...................

		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude
                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+30
				if i == 1:
					ln_param7 = salt+30
				elif i == len(x):
					ln_param7 = salt+30
				elif i == len(x)-1:
					ln_param7 = salt+30
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100

					else:
						ln_param7 = salt+30

			else:
				if i == 0:
					ln_param7 = salt
				elif i == len(x):
					ln_param7 = salt
				elif i == len(x)-1:
					ln_param7 = salt
				else:
					ln_param7 = 100
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_4.append(cmd)
		print ("....uav4...grid_wp..", wayPoints_lat_lon_uav4)
		missionlist_5_all.append(missionlist_4)


		#########..part 2.........

		#......................square 5(333x600)...................

		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude
                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+40
				if i == 1:
					ln_param7 = salt+40
				elif i == len(x):
					ln_param7 = salt+40
				elif i == len(x)-1:
					ln_param7 = salt+40
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100

					else:
						ln_param7 = salt+40

			else:
				if i == 0:
					ln_param7 = salt+40
				elif i == len(x):
					ln_param7 = salt+40
				elif i == len(x)-1:
					ln_param7 = salt+40
				else:
					ln_param7 = 100

			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_5.append(cmd)
		print ("....uav5...grid_wp..", wayPoints_lat_lon_uav5)
		missionlist_5_all.append(missionlist_5)

		#............................
                follower_host_tuple_heal = []
		for j, iter_follower_heal in enumerate(follower_host_tuple): 
			print (".....j.....", j)
			if int(self_heal[j]) > 0:
				print (">>>>>..lost uav for swarm")
			else:
				print ("....ok....")
				follower_host_tuple_heal.append(iter_follower_heal)

		#........................
		print ("...follower_host_tuple_heal....", follower_host_tuple_heal)
		for j, iter_follower in enumerate(follower_host_tuple_heal): 
			print (".....jjj...", j)
			cmdline(iter_follower,missionlist_5_all[j])
			print ("...%%%%....")
		#.......................	



	if count_swarm == 4:

		Tdis_4 = sqrt(pow(search_area,2)+pow(int(search_area/4),2))
		angle_4 = math.degrees(math.atan2(search_area, int(search_area/4)))

		print ("Tdis_4", Tdis_4)
		print ("angle_4",angle_4)
		#......................square 1(500x500)...................
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude
                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt
				if i == 1:
					ln_param7 = salt
				elif i == len(x):
					ln_param7 = salt
				elif i == len(x)-1:
					ln_param7 = salt
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100

					else:
						ln_param7 = salt

			else:
				if i == 0:
					ln_param7 = salt
				elif i == len(x):
					ln_param7 = salt
				elif i == len(x)-1:
					ln_param7 = salt
				else:
					ln_param7 = 100

				#ln_param7 = 300
			#cmd = Command( 0, 0, 0, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4, ln_param5, ln_param6, ln_param7)
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_1.append(cmd)
		print ("....uav1...grid_wp..", wayPoints_lat_lon_uav1)
		print ("....leng...", len(wayPoints_lat_lon_uav1))
		missionlist_4_all.append(missionlist_1)
		
	#......................square 2(500x500)...................
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude
                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+10
				if i == 1:
					ln_param7 = salt+10

				elif i == len(x):
					ln_param7 = salt+10
				elif i == len(x)-1:
					ln_param7 = salt+10
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100

					else:
						ln_param7 = salt+10

			else:
				if i == 0:
					ln_param7 = salt+10
				elif i == len(x):
					ln_param7 = salt+10
				elif i == len(x)-1:
					ln_param7 = salt+10
				else:
					ln_param7 = 100

			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_2.append(cmd)
		print ("....uav2...grid_wp..", wayPoints_lat_lon_uav2)
		missionlist_4_all.append(missionlist_2)

		#.................part 2.....................

		#......................square 3(500x500)...................
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude
                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+20
				if i == 1:
					ln_param7 = salt+20
				elif i == len(x):
					ln_param7 = salt+20
				elif i == len(x)-1:
					ln_param7 = salt+20
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100

					else:
						ln_param7 = salt

			else:
				if i == 0:
					ln_param7 = salt+20
				elif i == len(x):
					ln_param7 = salt+20
				elif i == len(x)-1:
					ln_param7 = salt+20
				else:
					ln_param7 = 100

			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_3.append(cmd)
		print ("....uav3...grid_wp..", wayPoints_lat_lon_uav3)
		missionlist_4_all.append(missionlist_3)

		#......................square 4(500x500)...................
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude
                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+30
				if i == 1:
					ln_param7 = salt+30
				elif i == len(x):
					ln_param7 = salt+30
				elif i == len(x)-1:
					ln_param7 = salt+30
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100

					else:
						ln_param7 = salt+30

			else:
				if i == 0:
					ln_param7 = salt+30
				elif i == len(x):
					ln_param7 = salt+30
				elif i == len(x)-1:
					ln_param7 = salt+30
				else:
					ln_param7 = 100

			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_4.append(cmd)
		print ("....uav4...grid_wp..", wayPoints_lat_lon_uav4)
		missionlist_4_all.append(missionlist_4)


		#........................
		#............................
                follower_host_tuple_heal = []
		for j, iter_follower_heal in enumerate(follower_host_tuple): 
			print (".....j.....", j)
			if int(self_heal[j]) > 0:
				print (">>>>>..lost uav for swarm")
			else:
				print ("....ok....")
				follower_host_tuple_heal.append(iter_follower_heal)

		#........................
		print ("...follower_host_tuple_heal....", follower_host_tuple_heal)
		for j, iter_follower in enumerate(follower_host_tuple_heal): 
			print (".....jjj...", j)
			cmdline(iter_follower,missionlist_4_all[j])
			print ("...%%%%....")
		#.......................	

		
		print ("....wp uploading is done ..")
		print ("all uav changed to to search")
	#..................................................
	if count_swarm == 3:
		Tdis_3 = sqrt(pow(search_area,2)+pow(int(search_area/3),2))
		angle_3 = math.degrees(math.atan2(search_area, int(search_area/3)))
		print ("Tdis_3", Tdis_3)
		print ("angle_3",angle_3)
		#.....part 1.....
		#......................square 1(1000x300)...................
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude

                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt
				if i == 1:
					ln_param7 = salt
				elif i == len(x):
					ln_param7 = salt
				elif i == len(x)-1:
					ln_param7 = salt
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100

					else:
						ln_param7 = salt

			else:
				if i == 0:
					ln_param7 = salt
				elif i == len(x):
					ln_param7 = salt
				elif i == len(x)-1:
					ln_param7 = salt
				else:
					ln_param7 = 100

				#ln_param7 = 300
			#cmd = Command( 0, 0, 0, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4, ln_param5, ln_param6, ln_param7)
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_1.append(cmd)
		print ("....uav1...grid_wp..", wayPoints_lat_lon_uav1)
		print ("....leng...", len(wayPoints_lat_lon_uav1))
		missionlist_3_all.append(missionlist_1)
		
	#.......part 2..............
	#......................square 2(500x700)...................
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude
	                if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+10
				if i == 1:
					ln_param7 = salt+10
				elif i == len(x):
					ln_param7 = salt+10
				elif i == len(x)-1:
					ln_param7 = salt+10
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100

					else:
						ln_param7 = salt+10

			else:
				if i == 0:
					ln_param7 = salt+10
				elif i == len(x):
					ln_param7 = salt+10
				elif i == len(x)-1:
					ln_param7 = salt+10
				else:
					ln_param7 = 100

			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_2.append(cmd)
		print ("....uav2...grid_wp..", wayPoints_lat_lon_uav2)
		missionlist_3_all.append(missionlist_2)

		#......................square 3(500x700)...................
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude

			if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+20
				if i == 1:
					ln_param7 = salt+20
				elif i == len(x):
					ln_param7 = salt+20
				elif i == len(x)-1:
					ln_param7 = salt+20
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100

					else:
						ln_param7 = salt+20

			else:
				if i == 0:
					ln_param7 = salt+20
				elif i == len(x):
					ln_param7 = salt+20
				elif i == len(x)-1:
					ln_param7 = salt+20
				else:
					ln_param7 = 100

			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_3.append(cmd)
		print ("....uav3...grid_wp..", wayPoints_lat_lon_uav3)
		missionlist_3_all.append(missionlist_3)

		#............................
                follower_host_tuple_heal = []
		for j, iter_follower_heal in enumerate(follower_host_tuple): 
			print (".....j.....", j)
			if int(self_heal[j]) > 0:
				print (">>>>>..lost uav for swarm")
			else:
				print ("....ok....")
				follower_host_tuple_heal.append(iter_follower_heal)

		#........................
		print ("...follower_host_tuple_heal....", follower_host_tuple_heal)
		for j, iter_follower in enumerate(follower_host_tuple_heal): 
			print (".....jjj...", j)
			cmdline(iter_follower,missionlist_3_all[j])
			print ("...%%%%....")
		#.......................	

#...............................
	if count_swarm == 2:
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude
                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt
				if i == 1:
					ln_param7 = salt
				elif i == len(x):
					ln_param7 = salt
				elif i == len(x)-1:
					ln_param7 = salt
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100

					else:
						ln_param7 = salt

			else:
				if i == 0:
					ln_param7 = salt
				elif i == len(x):
					ln_param7 = salt
				elif i == len(x)-1:
					ln_param7 = salt
				else:
					ln_param7 = 100
				#ln_param7 = 300
			#cmd = Command( 0, 0, 0, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4, ln_param5, ln_param6, ln_param7)
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_1.append(cmd)
		print ("....uav1...grid_wp..", wayPoints_lat_lon_uav1)
		print ("....leng...", len(wayPoints_lat_lon_uav1))
		missionlist_2_all.append(missionlist_1)
	
		
	#......................square 2(1000x500)...................
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude
                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt+10
				if i == 1:
					ln_param7 = salt+10
				elif i == len(x):
					ln_param7 = salt+10
				elif i == len(x)-1:
					ln_param7 = salt+10
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100

					else:
						ln_param7 = salt+10

			else:
				if i == 0:
					ln_param7 = salt+10
				elif i == len(x):
					ln_param7 = salt+10
				elif i == len(x)-1:
					ln_param7 = salt+10
				else:
					ln_param7 = 100
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_2.append(cmd)
		print ("....uav2...grid_wp..", wayPoints_lat_lon_uav2)
		missionlist_2_all.append(missionlist_2)

		#........................
		#............................
                follower_host_tuple_heal = []
		for j, iter_follower_heal in enumerate(follower_host_tuple): 
			print (".....j.....", j)
			if int(self_heal[j]) > 0:
				print (">>>>>..lost uav for swarm")
			else:
				print ("....ok....")
				follower_host_tuple_heal.append(iter_follower_heal)

		#........................
		print ("...follower_host_tuple_heal....", follower_host_tuple_heal)
		for j, iter_follower in enumerate(follower_host_tuple_heal): 
			print (".....jjj...", j)
			cmdline(iter_follower,missionlist_2_all[j])
			print ("...%%%%....")
		#.......................		
		print ("....wp uploading is done ..")
		print ("all uav changed to to search")


#...............................
	if count_swarm == 1:
		point_func = 'AB_point'
		survey_agri_grid_lat_lon = [(12.99621,80.18432), (12.995768,80.184589), (12.983979,80.153053), (12.984377,80.152864)]
		for m in range(0, len(survey_agri_grid_lat_lon)):
			geo_point_1 = survey_agri_grid_lat_lon[m]

			print geo_point_1
			x123, y123 = ImageUtils.GPStoImagePos(geo_point_1[0], geo_point_1[1], map_zoomLevel, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			pos_x_y = (x123, y123)
			ext_1.append(pos_x_y)

		holes = []
		if point_func == 'AB_point':
			polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=8)
		if point_func == 'multi_point': 
			#polygon = AreaPolygon(ext_1, (884, 536), angle_grid, interior=holes, ft=0.2)  #ft=0.2 more no.of grid point
			polygon = AreaPolygon(ext_1, (), angle_grid, interior=holes, ft=8)
		print(polygon.rtf.angle)
		ll = polygon.get_area_coverage()
		#print (ll)
		x, y = ll.xy
		print (x, y)
		print ("..???????...", len(x))			    
		output = 'QGC WPL 110\n'
        

		for i in range(0, len(x)):
			x_pos, y_pos = (x[i], y[i])
			print (x_pos, y_pos)
			latitude, longitude = ImageUtils.PostoGPS(int(x_pos*scale), int(y_pos*scale), grid_line_space, center_lat, center_lon, WIDTH_C, HEIGHT_C)


			print ("lat,lon", latitude,longitude)
			ln_param5, ln_param6 = latitude,longitude
                        if search_no_time == 1:                      
				if i == 0:
					ln_param7 = salt
				if i == 1:
					ln_param7 = salt
				elif i == len(x):
					ln_param7 = salt
				elif i == len(x)-1:
					ln_param7 = salt
				else:
					#ln_param7 = 300
					if search_no_time == 1:   #same alt search
						ln_param7 = 100

					else:
						ln_param7 = salt

			else:
				if i == 0:
					ln_param7 = salt
				elif i == len(x):
					ln_param7 = salt
				elif i == len(x)-1:
					ln_param7 = salt
				else:
					ln_param7 = 100

				#ln_param7 = 300
			#cmd = Command( 0, 0, 0, ln_frame, ln_command, ln_currentwp, ln_autocontinue, ln_param1, ln_param2, ln_param3, ln_param4, ln_param5, ln_param6, ln_param7)
			cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			missionlist_1.append(cmd)
		print ("....uav1...grid_wp..", wayPoints_lat_lon_uav1)
		print ("....leng...", len(wayPoints_lat_lon_uav1))
		missionlist_1_all.append(missionlist_1)

		#........................
		#............................
                follower_host_tuple_heal = []
		for j, iter_follower_heal in enumerate(follower_host_tuple): 
			print (".....j.....", j)
			if int(self_heal[j]) > 0:
				print (">>>>>..lost uav for swarm")
			else:
				print ("....ok....")
				follower_host_tuple_heal.append(iter_follower_heal)

		#........................
		print ("...follower_host_tuple_heal....", follower_host_tuple_heal)
		for j, iter_follower in enumerate(follower_host_tuple_heal): 
			print (".....jjj...", j)
			cmdline(iter_follower,missionlist_1_all[j])
			print ("...%%%%....")
		#.......................		

		
		print ("....wp uploading is done ..")
		print ("all uav changed to to search")
#..................................................		



def cmdline(vehicle,missionlist):
  if vehicle == None:
        print ("slave is lost")
  else:
    print("\nUpload mission from a file: %s" % export_mission_filename)
    print(' Clear mission')
    cmds = vehicle.commands
    cmds.clear()   

    #Add new mission to vehicle

    for command in missionlist:
        cmds.add(command)
   
    print('Upload mission to master')
    vehicle.commands.upload()  

def download_mission(vehicle):

    print(" Download mission from vehicle")
    missionlist=[]
    cmds = vehicle.commands
    cmds.download()
    cmds.wait_ready()
    for cmd in cmds:
        missionlist.append(cmd)
    return missionlist

def save_mission(vehicle, aFileName):

    print("\nSave mission from Vehicle to file: %s" % export_mission_filename)    
    #Download mission from vehicle
    missionlist = download_mission(vehicle)
    #Add file-format information
    output='QGC WPL 110\n'
    #Add home location as 0th waypoint
    home = vehicle.home_location
    output+="%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (0,1,0,16,0,0,0,0,home.lat,home.lon,home.alt,1)
    #Add commands
    for cmd in missionlist:
        commandline="%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (cmd.seq,cmd.current,cmd.frame,cmd.command,cmd.param1,cmd.param2,cmd.param3,cmd.param4,cmd.x,cmd.y,cmd.z,cmd.autocontinue)
	#print ("commandline", commandline)
        output+=commandline
    with open(aFileName, 'w') as file_:
        print(" Write mission to file")
        file_.write(output)

def altitude_inc(vehicle1, altitude):
    global search_flag, takeoff_flag
    """
    if takeoff_flag == True:
	print ("takeoff_flag flag")
    """
    if search_flag == True:
	print ("vehicle mode is AUTO")
    else:
	    if vehicle1 == None:
		    print ("slave is lost")
	    else:
		if altitude and altitude > 0.0 and vehicle1.armed:
		    print ("\n\tChanging vehicle1 altitude %0.1f meters!\n" % float(altitude))
		    vehicle1.mode = VehicleMode("GUIDED")
		    print (vehicle1.mode)
		    location = LocationGlobalRelative(vehicle1.location.global_frame.lat, vehicle1.location.global_frame.lon, float(altitude))
		    print (location)
		    try:
		        vehicle1.simple_goto(location)
		        return
		  
		    except KeyboardInterrupt:
		        print ("KeyInterrupt on Altitude change.")
		        pass
		else:
		    print ("\n\tInvalid altitude or vehicle1 not flying.\n")
		    pass

def aggregation_formation(vehicle,mode_v,target):
    global search_flag, takeoff_flag
    if takeoff_flag == True:
	print ("takeoff_flag flag")
    elif search_flag == True:
	print ("vehicle mode is AUTO")
    else:
	    if vehicle == None:
		print ("slave is lost")
	    else:
		#print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@slave is present")
		vehicle.mode = VehicleMode("GUIDED")
		#print ("mode", vehicle.mode.name)
		vehicle.mode = VehicleMode(mode_v)
		vehicle.simple_goto(target) 
		time.sleep(0.1)

def aggregation_formation_01(vehicle,mode_v,target, speed):
    if vehicle == None:
        print ("slave is lost")
    else:
        #print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@slave is present")
        vehicle.mode = VehicleMode("GUIDED")
        #print ("mode", vehicle.mode.name)
        vehicle.mode = VehicleMode(mode_v)
        vehicle.simple_goto(target, groundspeed=speed) 
        time.sleep(0.1)


def speed():
    global check_box_flag3       
    global xy_pos, latlon_pos
    global master, no_uavs
    global self_heal
    global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10,vehicle11,vehicle12,vehicle13
    global vehicle14,vehicle15,vehicle16,vehicle17,vehicle18,vehicle19,vehicle20,vehicle21,vehicle22,vehicle23,vehicle24,vehicle25      
    global follower_host_tuple_G1,follower_host_tuple_G2,follower_host_tuple_G3,follower_host_tuple_G4,follower_host_tuple_G5
    global counter_G1,counter_G2,counter_G3,counter_G4,counter_G5

    global follower_host_tuple
    global circle_pos_flag
    xoffset = xoffset_entry.get()
    cradius = cradius_entry.get()
    aoffset = aoffset_entry.get()
    salt = salt_entry.get() 
    xoffset = int(xoffset)
    cradius = int(cradius)
    aoffset = int(aoffset)
    salt = int(salt)
    speed = speedset_entry.get()
    speed = int(speed)
    print ("speed set", speed)  
    if checkboxvalue_Group_1.get() == 1:
	Group_1()
	alt_001 = salt
	for i, iter_follower_G1 in enumerate(follower_host_tuple_G1):
	    if iter_follower_G1 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		print ("payload present  uav :", (i+1)) 
		iter_follower_G1.airspeed = speed
		time.sleep(0.2)
    if checkboxvalue_Group_2.get() == 1:
	Group_2()
	alt_001 = salt
	for i, iter_follower_G2 in enumerate(follower_host_tuple_G2):
	    if iter_follower_G2 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		print ("payload present  uav :", (i+1)) 
		iter_follower_G2.airspeed = speed
		time.sleep(0.2)
    if checkboxvalue_Group_3.get() == 1:
	Group_3()
	alt_001 = salt
	for i, iter_follower_G3 in enumerate(follower_host_tuple_G3):
	    if iter_follower_G3 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		print ("payload present  uav :", (i+1)) 
		iter_follower_G3.airspeed = speed
		time.sleep(0.2)
    if checkboxvalue_Group_4.get() == 1:
	Group_4()
	alt_001 = salt
	for i, iter_follower_G4 in enumerate(follower_host_tuple_G4):
	    if iter_follower_G4 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		print ("payload present  uav :", (i+1)) 
		iter_follower_G4.airspeed = speed
		time.sleep(0.2)

    if checkboxvalue_Group_all.get() == 1:
	    for iter_follower in follower_host_tuple: 
		if iter_follower == None:
		    print ("slave is lost")
		else:       
		    iter_follower.airspeed = speed
		    time.sleep(0.2)



def Remove():
        global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10, vehicle11, vehicle12, vehicle13, vehicle14,vehicle15
        global self_heal
        global RTH_array
        global follower_host_tuple
        Remove_port = Removeset_entry.get()
        #Reconnect = int(Reconnect)
        print ("Remove set", Remove_port)  
        port = ['14551', '14552', '14553','14554','14555','14556','14557','14558','14559', '14560', '14561', '14562', '14563', '14564', '14565', '14566', '14567', '14568', '14569', '14570', '14571', '14572', '14573', '14574', '14575']
        for i,iter_follower in enumerate(port):
            if str(Remove_port) == iter_follower:
                try:
                    if i == 0:
                        #vehicle1.close()
                        #vehicle1 = None
                        self_heal[0] = 1
                        RTH_array[0] = 1
                        data1_ = Label(text = "UAV_01", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 40)
                        #count1 = 0     
                    elif i == 1:
                        #vehicle2.close()
                        #vehicle2 = None
                        self_heal[1] = 2
                        RTH_array[1] = 1
                        data2_ = Label(text = "UAV_02", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 75)
                        #count2 = 0     
                    elif i == 2:
                        #vehicle3.close()
                        #vehicle3 = None
                        self_heal[2] = 3
                        RTH_array[2] = 1
                        data3_ = Label(text = "UAV_03", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 110)
                        #count3 = 0     
                    elif i == 3:
                        #vehicle4.close()
                        #vehicle4 = None
                        self_heal[3] = 4
                        RTH_array[3] = 1
                        data4_ = Label(text = "UAV_04", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 145)
                        #count4 = 0             
                    elif i == 4:
                        #vehicle5.close()
                        #vehicle5 = None
                        self_heal[4] = 5
                        RTH_array[4] = 1
                        data5_ = Label(text = "UAV_05", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 180)
                        #count5 = 0         
                    elif i == 5:
                        #vehicle6.close()
                        print ("vehicle 6 is removed............")
                        #vehicle6 = None
                        self_heal[5] = 6
                        RTH_array[5] = 1
                        data6_ = Label(text = "UAV_06", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 215)
                        #count6 = 0     

                    elif i == 6:
                        #vehicle7.close()
                        #vehicle7 = None
                        self_heal[6] = 7
                        RTH_array[6] = 1
                        data7_ = Label(text = "UAV_07", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 250) 
                        #count7 = 0     
                    elif i == 7:
                        #vehicle8.close()
                        #vehicle8 = None
                        self_heal[7] = 8 
                        RTH_array[7] = 1  
                        data8_ = Label(text = "UAV_08", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 285) 
                        #count8 = 0     
                    elif i == 8:
                        #vehicle9.close()
                        #vehicle9 = None
                        self_heal[8] = 9    
                        data9_ = Label(text = "UAV_09", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 320) 
                        #count9 = 0     
                    elif i == 9:
                        #vehicle10.close()
                        #vehicle10 = None
                        self_heal[9] = 10
                        data10_ = Label(text = "UAV_10", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 355)    
                        #count10 = 0      
                    elif i == 10:
                        #vehicle1.close()
                        #vehicle1 = None
                        self_heal[10] = 11
                        data1_ = Label(text = "UAV_11", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 390)
                        #count1 = 0     
                    elif i == 11:
                        #vehicle2.close()
                        #vehicle2 = None
                        self_heal[11] = 12
                        data2_ = Label(text = "UAV_12", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 425)
                        #count2 = 0     
                    elif i == 12:
                        #vehicle3.close()
                        #vehicle3 = None
                        self_heal[12] = 13
                        data3_ = Label(text = "UAV_13", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 460)
                        #count3 = 0     
                    elif i == 13:
                        #vehicle4.close()
                        #vehicle4 = None
                        self_heal[13] = 14
                        data4_ = Label(text = "UAV_14", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 495)
                        #count4 = 0             
                    elif i == 14:
                        #vehicle5.close()
                        #vehicle5 = None
                        self_heal[14] = 15
                        data5_ = Label(text = "UAV_15", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 530)
                        #count5 = 0         
                    elif i == 15:
                        #vehicle6.close()
                        print ("vehicle 6 is removed............")
                        #vehicle6 = None
                        self_heal[15] = 16
                        data6_ = Label(text = "UAV_16", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 565)
                        #count6 = 0     

                    elif i == 16:
                        #vehicle7.close()
                        #vehicle7 = None
                        self_heal[16] = 17
                        data7_ = Label(text = "UAV_17", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 600) 
                        #count7 = 0     
                    elif i == 17:
                        #vehicle8.close()
                        #vehicle8 = None
                        self_heal[17] = 18   
                        data8_ = Label(text = "UAV_18", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 635) 
                        #count8 = 0     
                    elif i == 18:
                        #vehicle9.close()
                        #vehicle9 = None
                        self_heal[18] = 19    
                        data9_ = Label(text = "UAV_19", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 670) 
                        #count9 = 0     
                    elif i == 19:
                        #vehicle10.close()
                        #vehicle10 = None
                        self_heal[19] = 20
                        data10_ = Label(text = "UAV_20", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 705)    
                        #count10 = 0       
                    elif i == 20:
                        #vehicle1.close()
                        #vehicle1 = None
                        self_heal[20] = 21
                        data1_ = Label(text = "UAV_21", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 740)
                        #count1 = 0     
                    elif i == 21:
                        #vehicle2.close()
                        #vehicle2 = None
                        self_heal[21] = 22
                        data2_ = Label(text = "UAV_22", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 775)
                        #count2 = 0     
                    elif i == 22:
                        #vehicle3.close()
                        #vehicle3 = None
                        self_heal[22] = 23
                        data3_ = Label(text = "UAV_23", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 810)
                        #count3 = 0     
                    elif i == 23:
                        #vehicle4.close()
                        #vehicle4 = None
                        self_heal[23] = 24
                        data4_ = Label(text = "UAV_24", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 845)
                        #count4 = 0             
                    elif i == 24:
                        #vehicle5.close()
                        #vehicle5 = None
                        self_heal[24] = 25
                        data5_ = Label(text = "UAV_25", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 880)
                        #count5 = 0         
 
          

                except:
                    pass
    

def Reconnect():
    global self_heal, self_heal
    global follower_host_tuple
    global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10,vehicle11,vehicle12,vehicle13
    global vehicle14,vehicle15,vehicle16,vehicle17,vehicle18,vehicle19,vehicle20,vehicle21,vehicle22,vehicle23,vehicle24,vehicle25
    global count1,count2,count3,count4,count5,count6,count7,count8,count9,count10,count11,count12,count13,count14,count15,count16,count17,count18,count19,count20
    global count21,count22,count23,count24,count25
    global RTH_array

    Reconnect_port = Reconnectset_entry.get()
    #Reconnect = int(Reconnect)
    print ("Reconnect set", Reconnect_port)  
    connection_string = "udpin:192.168.6.210:"+str(Reconnect_port)
    port = ['14551', '14552', '14553','14554','14555','14556','14557','14558','14559', '14560', '14561', '14562', '14563', '14564', '14565']

    for i,iter_follower in enumerate(port):
    	print("iter_follower",iter_follower)
        if str(Reconnect_port) == iter_follower:
            print("^^^^^^^^^^^",i)
            try:
            	vehicle = connect(connection_string, baud=115200, heartbeat_timeout=30)
                print("vehicle",vehicle)
                vehicle.wait_ready('autopilot_version')
                time.sleep(0.1)
                follower_host_tuple[i] = vehicle

                if i == 0:
                    follower_host_tuple[0] = vehicle
                    print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv1")
                    vehicle1 = vehicle
                    self_heal[0] = 0
		    RTH_array[0] = 100
                    data1_ = Label(text = "UAV_01", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 40)
                    count1 = 0      

                elif i == 1:
                    follower_host_tuple[1] = vehicle
                    print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv3")
                    vehicle2 = vehicle
                    self_heal[1] = 0
		    RTH_array[1] = 100
                    data2_ = Label(text = "UAV_02", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 75)
                    count2 = 0      

                elif i == 2:
                    follower_host_tuple[2] = vehicle
                    print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv5")
                    vehicle3 = vehicle
                    self_heal[2] = 0
		    RTH_array[2] = 100
                    data3_ = Label(text = "UAV_03", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 110)
                    count3 = 0          

                elif i == 3:
                    follower_host_tuple[3] = vehicle
                    print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv7")
                    vehicle4 = vehicle  
                    self_heal[3] = 0
		    RTH_array[3] = 100
                    data4_ = Label(text = "UAV_04", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 145)   
                    count4 = 0      

                elif i == 4:
                    follower_host_tuple[4] = vehicle
                    print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv9")
                    vehicle5 = vehicle
                    self_heal[4] = 0  
		    RTH_array[4] = 100  
                    data5_ = Label(text = "UAV_05", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 180)   
                    count5 = 0      

                elif i == 5:
                    follower_host_tuple[5] = vehicle
                    print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv1")
                    vehicle6 = vehicle
                    self_heal[5] = 0
		    RTH_array[5] = 100
                    data6_ = Label(text = "UAV_06", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 215)
                    count6 = 0      

                elif i == 6:
                    follower_host_tuple[6] = vehicle
                    print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv3")
                    vehicle7 = vehicle
                    self_heal[6] = 0
		    RTH_array[6] = 100
                    data7_ = Label(text = "UAV_07", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 250)
                    count7 = 0      

                elif i == 7:
                    print("*****************")
                    follower_host_tuple[7] =vehicle
                    print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv51")
                    vehicle8 = vehicle
                    print("vehicle8",vehicle8)
                    self_heal[7] = 0
                    print("*****00000******")
		    RTH_array[7] = 100
		    print("*****111111******")
                    data8_ = Label(text = "UAV_08", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 285)
                    print("*****222222******")
                    count8 = 0          

                elif i == 8:
                    follower_host_tuple[8] = vehicle
                    print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv7")
                    vehicle9 = vehicle  
                    self_heal[8] = 0
                    data9_ = Label(text = "UAV_09", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 320)   
                    count9 = 0      

                elif i == 9:
                    follower_host_tuple[9] = vehicle
                    print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv9")
                    vehicle10 = vehicle
                    self_heal[9] = 0    
                    data10_ = Label(text = "UAV_10", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 355)   
                    count10 = 0     
                elif i == 10:
                    follower_host_tuple[10] = vehicle
                    print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv1")
                    vehicle11 = vehicle
                    self_heal[10] = 0
                    data1_ = Label(text = "UAV_11", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 390)
                    count1 = 0      

                elif i == 11:
                    follower_host_tuple[11] = vehicle
                    print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv3")
                    vehicle12 = vehicle
                    self_heal[11] = 0
                    data2_ = Label(text = "UAV_12", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 425)
                    count2 = 0      

                elif i == 12:
                    follower_host_tuple[12] = vehicle
                    print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv5")
                    vehicle13 = vehicle
                    self_heal[12] = 0
                    data3_ = Label(text = "UAV_13", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 460)
                    count3 = 0          

                elif i == 13:
                    follower_host_tuple[13] = vehicle
                    print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv7")
                    vehicle14 = vehicle  
                    self_heal[13] = 0
                    data4_ = Label(text = "UAV_14", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 495)   
                    count4 = 0      

                elif i == 14:
                    follower_host_tuple[14] = vehicle
                    print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv9")
                    vehicle15 = vehicle
                    self_heal[14] = 0    
                    data5_ = Label(text = "UAV_15", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 530)   
                    count5 = 0      

                elif i == 15:
                    follower_host_tuple[15] = vehicle
                    print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv1")
                    vehicle16 = vehicle
                    self_heal[15] = 0
                    data6_ = Label(text = "UAV_16", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 565)
                    count6 = 0      

                elif i == 16:
                    follower_host_tuple[16] = vehicle
                    print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv3")
                    vehicle17 = vehicle
                    self_heal[16] = 0
                    data7_ = Label(text = "UAV_17", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 600)
                    count7 = 0      

                elif i == 17:
                    follower_host_tuple[17] = vehicle
                    print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv5")
                    vehicle18 = vehicle
                    self_heal[17] = 0
                    data8_ = Label(text = "UAV_18", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 635)
                    count8 = 0          

                elif i == 18:
                    follower_host_tuple[18] = vehicle
                    print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv7")
                    vehicle19 = vehicle  
                    self_heal[18] = 0
                    data9_ = Label(text = "UAV_19", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 670)   
                    count9 = 0      

                elif i == 19:
                    follower_host_tuple[19] = vehicle
                    print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv9")
                    vehicle20 = vehicle
                    self_heal[19] = 0    
                    data10_ = Label(text = "UAV_20", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 705)   
                    count10 = 0  
                elif i == 20:
                    follower_host_tuple[20] = vehicle
                    print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv1")
                    vehicle21 = vehicle
                    self_heal[20] = 0
                    data1_ = Label(text = "UAV_21", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 740)
                    count1 = 0      

                elif i == 21:
                    follower_host_tuple[21] = vehicle
                    print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv3")
                    vehicle22 = vehicle
                    self_heal[21] = 0
                    data2_ = Label(text = "UAV_22", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 775)
                    count2 = 0      

                elif i == 22:
                    follower_host_tuple[22] = vehicle
                    print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv5")
                    vehicle23 = vehicle
                    self_heal[22] = 0
                    data3_ = Label(text = "UAV_23", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 810)
                    count3 = 0          

                elif i == 23:
                    follower_host_tuple[23] = vehicle
                    print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv7")
                    vehicle24 = vehicle  
                    self_heal[23] = 0
                    data4_ = Label(text = "UAV_24", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 845)   
                    count4 = 0      

                elif i == 24:
                    follower_host_tuple[24] = vehicle
                    print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv9")
                    vehicle25 = vehicle
                    self_heal[24] = 0    
                    data5_ = Label(text = "UAV_25", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 880)   
                    count5 = 0      

        
            except:
                print ("vehicle is not able reconnect")



    print ("reconnect uav connection check complete")
    print ("follower_host_tuple", follower_host_tuple)


def skip():
    global follower_host_tuple
    skip1 = skip1set_entry.get()
    skip1 = int(skip1)
    print ("skip1 set", skip1)              
    for iter_follower_main in follower_host_tuple:            
        if iter_follower_main == None:
            print ("slave is lost")
        else:
            nextwaypoint = iter_follower_main.commands.next
            print ("nextwaypoint", nextwaypoint)
            iter_follower_main.mode = VehicleMode('AUTO')
            time.sleep(0.2)
            iter_follower_main.commands.next = int(skip1)
            time.sleep(0.1)

def skip1():
    global follower_host_tuple_main
    skip1 = skip1set_entry.get()
    skip1 = int(skip1)
    print ("skip1 set", skip1)              
    for iter_follower_main in follower_host_tuple_main:
            
        if iter_follower_main == None:
            print ("slave is lost")
        else:
            nextwaypoint = iter_follower_main.commands.next
            print ("nextwaypoint", nextwaypoint)
            iter_follower_main.mode = VehicleMode('AUTO')
            time.sleep(0.2)
            iter_follower_main.commands.next = int(skip1)
            time.sleep(0.1)


def skip2():
    global follower_host_tuple_sec
    skip2 = skip2set_entry.get()
    skip2 = int(skip2)
    print ("skip2 set", skip2)              
    for iter_follower_sec in follower_host_tuple_sec:
            
        if iter_follower_sec == None:
            print ("slave is lost")
        else:
            nextwaypoint = iter_follower_sec.commands.next
            print ("nextwaypoint", nextwaypoint)
            iter_follower_sec.mode = VehicleMode('AUTO')
            time.sleep(0.2)
            iter_follower_sec.commands.next = int(skip2)
            time.sleep(0.1)
'''
def master():
    global master
    master = masterset_entry.get()
    master = int(master)
    print ("master set", master)      
'''

def master():
    global master_ip,master,socket1,file_sock,server_address1,server_address11,file_server_address,follower_host_tuple,server_address12,socket2
    master = masterset_entry.get()
    master = int(master)
    print ("master set", master)
    #msg="uav_num"+","str(len(follower_host_tuple))+","+"master"+","+str(master)
    data = "master" + "-" + str(master)
    print("data",data)
    '''
    socket1.sendto(str(data).encode(), server_address11)
    time.sleep(0.5)
    socket2.sendto(str(data).encode(), server_address12)
    time.sleep(0.5)
    '''
    socket3.sendto(str(data).encode(), server_address13)
    time.sleep(0.5)
    socket4.sendto(str(data).encode(), server_address14)
    time.sleep(0.5)
    socket5.sendto(str(data).encode(), server_address15)
    time.sleep(0.5)
    '''
    socket6.sendto(str(data).encode(), server_address16)
    time.sleep(0.5)
    socket7.sendto(str(data).encode(), server_address17)
    time.sleep(0.5)
    socket8.sendto(str(data).encode(), server_address18)
    time.sleep(0.5)
    socket9.sendto(str(data).encode(), server_address19)
    time.sleep(0.5)
    '''
    socket10.sendto(str(data).encode(), server_address20)
    if master==1:
    	master_ip='192.168.6.151'
    elif master==2:
    	master_ip='192.168.6.152'
    elif master==3:
    	master_ip='192.168.6.153'
    elif master==4:
    	master_ip='192.168.6.154'
    elif master==5:
    	master_ip='192.168.6.155'
    elif master==6:
    	master_ip='192.168.6.156'
    elif master==7:
    	master_ip='192.168.6.157'
    elif master==8:
    	master_ip='192.168.6.158'
    elif master==9:
    	master_ip='192.168.6.159'
    elif master==10:
    	master_ip='192.168.6.160'
    mavlink_add()

def mavlink_add():
    global master,master_ip
    master_ip_array=['192.168.6.151','192.168.6.152','192.168.6.153','192.168.6.154','192.168.6.155','192.168.6.156','192.168.6.157','192.168.6.158','192.168.6.159','192.168.6.160']
    master_ip=master_ip_array[int(master)-1]
    print("master_ip",master_ip)
    add='output add '
    remove='output remove '

    port_array=[':14551',':14552',':14553',':14554',':14555',':14556',':14557',':14558',':14559',':14560']

    
    data=add+master_ip
    #data=remove+master_ip
    print("data",(data+port_array[0]),type(data))	
    '''
    mavlink_sock1.sendto((data+port_array[0]).encode(), mavlink_server_address1)
    time.sleep(0.5)
    mavlink_sock2.sendto((data+port_array[1]).encode(), mavlink_server_address2)
    time.sleep(0.5)
    '''
    mavlink_sock3.sendto((data+port_array[2]).encode(), mavlink_server_address3)
    time.sleep(0.5)
    mavlink_sock4.sendto((data+port_array[3]).encode(), mavlink_server_address4)
    time.sleep(0.5)
    mavlink_sock5.sendto((data+port_array[4]).encode(), mavlink_server_address5)
    time.sleep(0.5)
    '''
    mavlink_sock6.sendto((data+port_array[5]).encode(), mavlink_server_address6)
    time.sleep(0.5)
    
    mavlink_sock7.sendto((data+port_array[6]).encode(), mavlink_server_address7)
    time.sleep(0.5)
    mavlink_sock8.sendto((data+port_array[7]).encode(), mavlink_server_address8)
    time.sleep(0.5)
    mavlink_sock9.sendto((data+port_array[8]).encode(), mavlink_server_address9)
    time.sleep(0.5)
    '''
    mavlink_sock10.sendto((data+port_array[9]).encode(), mavlink_server_address10)

def mavlink_remove():
    global master,master_ip
    #data = removelink_entry.get()
    master_ip_array=['192.168.6.151','192.168.6.152','192.168.6.153','192.168.6.154','192.168.6.155','192.168.6.156','192.168.6.157','192.168.6.158','192.168.6.159','192.168.6.160']
    remove_link=removelink_entry.get()
    print("remove_link",remove_link)
    master_ip=master_ip_array[int(remove_link)-1]
    print("master_ip",master_ip)
    add='output add '
    remove='output remove '
    port_array=[':14551',':14552',':14553',':14554',':14555',':14556',':14557',':14558',':14559',':14560']
    data=remove+master_ip
    print("data",(data+port_array[0]),type(data))
    '''
    mavlink_sock1.sendto((data+port_array[0]).encode(), mavlink_server_address1)
    time.sleep(0.5)
    mavlink_sock2.sendto((data+port_array[1]).encode(), mavlink_server_address2)
    time.sleep(0.5)
    '''
    mavlink_sock3.sendto((data+port_array[2]).encode(), mavlink_server_address3)
    time.sleep(0.5)
    mavlink_sock4.sendto((data+port_array[3]).encode(), mavlink_server_address4)
    time.sleep(0.5)
    mavlink_sock5.sendto((data+port_array[4]).encode(), mavlink_server_address5)
    time.sleep(0.5)
    '''
    mavlink_sock6.sendto((data+port_array[5]).encode(), mavlink_server_address6)
    time.sleep(0.5)
    mavlink_sock7.sendto((data+port_array[6]).encode(), mavlink_server_address7)
    time.sleep(0.5)
    mavlink_sock8.sendto((data+port_array[7]).encode(), mavlink_server_address8)
    time.sleep(0.5)
    mavlink_sock9.sendto((data+port_array[8]).encode(), mavlink_server_address9)
    time.sleep(0.5)
    '''
    mavlink_sock10.sendto((data+port_array[9]).encode(), mavlink_server_address10)
	
def bot_remove():
        print("!!!bot_remove!!")
        remove_bot_num=removebot_entry.get()
        print("remove_link_num",remove_bot_num)
        global udp_socket,udp_socket2,server_address1,server_address2
        #udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data = "remove_bot"+","+str(remove_bot_num)
        print("remove_link_num",remove_bot_num)
        '''
        udp_socket.sendto(str(data).encode(), server_address1)
        time.sleep(0.5)
        udp_socket2.sendto(str(data).encode(), server_address2)
        time.sleep(0.5)
	'''
        udp_socket3.sendto(str(data).encode(), server_address3)
        time.sleep(0.5)
        udp_socket4.sendto(str(data).encode(), server_address4)
        time.sleep(0.5)
        udp_socket5.sendto(str(data).encode(), server_address5)
        time.sleep(0.5)
        '''
        udp_socket6.sendto(str(data).encode(), server_address6)
        time.sleep(0.5)
        udp_socket7.sendto(str(data).encode(), server_address7)
        time.sleep(0.5)
        udp_socket8.sendto(str(data).encode(), server_address8)
        time.sleep(0.5)
        udp_socket9.sendto(str(data).encode(), server_address9)
        time.sleep(0.5)
        '''
        udp_socket10.sendto(str(data).encode(), server_address10)

	
def no_uavs():
    global no_uavs
    no_uavs = no_uavsset_entry.get()
    no_uavs = int(no_uavs)
    print ("no_uavs set", no_uavs)   
"""
def Target_upload():

    global loc_01
    global follower_host_tuple
    loc_01 = []
    with open('data.csv','rt')as text:
        data = csv.reader(text)
        #total_data = list(data)
        #total_target = len(total_data) 
        #print "total_target foundmm", total_target
        print (data)
        m = 0
        for row in data:
            if m != 0:
                data = (row)
                print ("loc", data[0], data[1])
                tar_data= [float(data[0]), float(data[1])] 
                loc_01.append(tar_data) 
                #target = LocationGlobalRelative(data[0], data[1], alt)
                #print ("target", target)
                #aggregation_formation(iter_follower,"GUIDED", target)   
            m = m+1
        print ("loc_01", loc_01)
        for k, iter_follower in enumerate(follower_host_tuple): 
            if iter_follower == None:
                print ("slave is lost")
            else:     
                if k > len(loc_01):
                    print ('k', k)
                    break 
                alt_123 = iter_follower.location.global_relative_frame.alt
                target = LocationGlobalRelative(loc_01[k][0], loc_01[k][1], alt_123)
                aggregation_formation_01(iter_follower,"GUIDED", target, 8) 
"""
"""
def Home_pos():
	print ("#.......!!!!................all uav move to home position")
        global xy_pos, check_box_flag3
        global master, no_uavs
        global self_heal
        global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10, vehicle11, vehicle12, vehicle13, vehicle14,vehicle15,vehicle16,vehicle17,vehicle18,vehicle19,vehicle20,vehicle21,vehicle22,vehicle23,vehicle24,vehicle25
        global follower_host_tuple
        global circle_pos_flag

        xoffset = xoffset_entry.get()
        cradius = cradius_entry.get()
        aoffset = aoffset_entry.get()
        salt = salt_entry.get() 
        xoffset = int(xoffset)
        cradius = int(cradius)
        aoffset = int(aoffset)
        salt = int(salt)
        print ("aggregation", master)

        if master == 1:
            lat = vehicle1.location.global_relative_frame.lat
            lon = vehicle1.location.global_relative_frame.lon

        if master == 2:
            lat = vehicle2.location.global_relative_frame.lat
            lon = vehicle2.location.global_relative_frame.lon

        if master == 3:
            lat = vehicle3.location.global_relative_frame.lat
            lon = vehicle3.location.global_relative_frame.lon
                 
        if master == 4:
            lat = vehicle4.location.global_relative_frame.lat
            lon = vehicle4.location.global_relative_frame.lon

        if master == 5:
            lat = vehicle5.location.global_relative_frame.lat
            lon = vehicle5.location.global_relative_frame.lon

        if master == 6:
            lat = vehicle6.location.global_relative_frame.lat
            lon = vehicle6.location.global_relative_frame.lon

        if master == 7:
            lat = vehicle7.location.global_relative_frame.lat
            lon = vehicle7.location.global_relative_frame.lon

        if master == 8:
            lat = vehicle8.location.global_relative_frame.lat
            lon = vehicle8.location.global_relative_frame.lon

        if master == 9:
            lat = vehicle9.location.global_relative_frame.lat
            lon = vehicle9.location.global_relative_frame.lon

        if master == 10:
            lat = vehicle10.location.global_relative_frame.lat
            lon = vehicle10.location.global_relative_frame.lon

        if checkboxvalue1.get() == 1:
            formation(int(no_uavs),  'T', lat, lon)
        elif checkboxvalue2.get() == 1:
            formation(int(no_uavs),  'L', lat, lon)
        elif checkboxvalue3.get() == 1:
            formation(int(no_uavs), 'S', lat, lon)
        elif checkboxvalue4.get() == 1:
            if circle_pos_flag == True:
                circle_pos_flag = False
                clat = clat_entry.get()
                clon = clon_entry.get()
                clat = float(clat)
                clon = float(clon)
                formation(int(no_uavs), 'C', clat, clon)
            else:
                formation(int(no_uavs), 'C', lat, lon)


        a,b,c = (0,0,0)
        count_wp = 0
        print (follower_host_tuple)
        print ("................", self_heal)
        #for i in range(0, int(no_uavs)):  
        alt_001 = salt
        for i, iter_follower in enumerate(follower_host_tuple): 
            if self_heal[i] > 0:
                ##print "lost odd uav", self_heal[i]
                print ("lost  uav", (i+1))
                if check_box_flag3 == True: 
                	print ("self heal..to alt change")   
		else:
                	alt_001 = alt_001 + aoffset   #....... alt not change during self heal...
  
            else: 
                #if i < int(no_uavs):   
                print ("present  uav :", (i+1)) 
		while not iter_follower.home_location:
			cmds = iter_follower.commands   
    			cmds.download()
    			cmds.wait_ready()
			iter_follower.home_location
			if not iter_follower.home_location:
				print ("waiting for home location")
			else:
				target = LocationGlobalRelative(iter_follower.home_location.lat, iter_follower.home_location.lon, alt_001)
				print ("target....home location..", target)
				for i in range(0, 5):
					print ("......home location....")
					aggregation_formation(iter_follower,"GUIDED", target)        
				alt_001 = alt_001 + aoffset

"""
def Home_pos():
	print ("#.......!!!!................all uav move to home position")
        global xy_pos, check_box_flag3
        global master, no_uavs
        global self_heal
        global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10
        global follower_host_tuple
        global circle_pos_flag
	global home_loc_all

        xoffset = xoffset_entry.get()
        cradius = cradius_entry.get()
        aoffset = aoffset_entry.get()
        salt = salt_entry.get() 
        xoffset = int(xoffset)
        cradius = int(cradius)
        aoffset = int(aoffset)
        salt = int(salt)
        print ("aggregation", master)

        if master == 1:
            lat = vehicle1.location.global_relative_frame.lat
            lon = vehicle1.location.global_relative_frame.lon

        if master == 2:
            lat = vehicle2.location.global_relative_frame.lat
            lon = vehicle2.location.global_relative_frame.lon

        if master == 3:
            lat = vehicle3.location.global_relative_frame.lat
            lon = vehicle3.location.global_relative_frame.lon
                 
        if master == 4:
            lat = vehicle4.location.global_relative_frame.lat
            lon = vehicle4.location.global_relative_frame.lon

        if master == 5:
            lat = vehicle5.location.global_relative_frame.lat
            lon = vehicle5.location.global_relative_frame.lon

        if master == 6:
            lat = vehicle6.location.global_relative_frame.lat
            lon = vehicle6.location.global_relative_frame.lon

        if master == 7:
            lat = vehicle7.location.global_relative_frame.lat
            lon = vehicle7.location.global_relative_frame.lon

        if master == 8:
            lat = vehicle8.location.global_relative_frame.lat
            lon = vehicle8.location.global_relative_frame.lon

        if master == 9:
            lat = vehicle9.location.global_relative_frame.lat
            lon = vehicle9.location.global_relative_frame.lon

        if master == 10:
            lat = vehicle10.location.global_relative_frame.lat
            lon = vehicle10.location.global_relative_frame.lon

        if checkboxvalue1.get() == 1:
            formation(int(no_uavs),  'T', lat, lon)
        elif checkboxvalue2.get() == 1:
            formation(int(no_uavs),  'L', lat, lon)
        elif checkboxvalue3.get() == 1:
            formation(int(no_uavs), 'S', lat, lon)
        elif checkboxvalue4.get() == 1:
            if circle_pos_flag == True:
                circle_pos_flag = False
                clat = clat_entry.get()
                clon = clon_entry.get()
                clat = float(clat)
                clon = float(clon)
                formation(int(no_uavs), 'C', clat, clon)
            else:
                formation(int(no_uavs), 'C', lat, lon)
        original_location = [lat, lon]

        a,b,c = (0,0,0)
        count_wp = 0
        print (follower_host_tuple)
        print ("................", self_heal)
        #for i in range(0, int(no_uavs)):  
        alt_001 = salt
        loc_home = []
        with open('myfile.csv','rt')as text:
		data = csv.reader(text)
		print "ddd"
		for row in data: 
			data = (row)
			loc_home.append(data)
        print ("loc_home", loc_home)
        for i, iter_follower in enumerate(follower_host_tuple): 
            if self_heal[i] > 0:
                ##print "lost odd uav", self_heal[i]
                print ("lost  uav", (i+1))
                if check_box_flag3 == True: 
                	print ("self heal..to alt change")   
		else:
                	alt_001 = alt_001 + aoffset   #....... alt not change during self heal...
  
            else: 
                #if i < int(no_uavs):   
                print ("present  uav :", (i+1)) 
                """
		home_pos_lat = home_loc_all[i][0]
		home_pos_lon = home_loc_all[i][1]
		"""
                print ("..loc_home[i]", loc_home[i])
		home_pos_lat = float(loc_home[i][0])
		home_pos_lon = float(loc_home[i][1])

                print ("..home_pos_lat....", home_pos_lat)
                print ("..home_pos_lon....", home_pos_lon)

		target = LocationGlobalRelative(home_pos_lat, home_pos_lon, iter_follower.location.global_relative_frame.alt)
		print ("target....home location..", target)
		for i in range(0, 5):
			print ("......home location....")
			aggregation_formation(iter_follower,"GUIDED", target)        
		alt_001 = alt_001 + aoffset


'''
def home_lock():
        global check_box_flag3       
        global xy_pos, latlon_pos
        global master, no_uavs
        global self_heal
        global follower_host_tuple
	global home_loc_all
	home_loc_all = []
	Target_csv_01 = open('myfile.csv', 'wb')
        for i, iter_follower in enumerate(follower_host_tuple):
	    if self_heal[i] > 0:
		print ("....lost uavs...") 

            else: 
		
		lat = iter_follower.location.global_relative_frame.lat
		lon = iter_follower.location.global_relative_frame.lon  

                print ("lat,lon", lat,lon)
		original_location = (lat, lon) 
		#home_loc_all.append(original_location)

		RESULT=[[lat, lon]]
		wr = csv.writer(Target_csv_01, dialect='excel')
		wr.writerows(RESULT)  

                print (" present  uav :", (i+1))    
        print ("home_loc_all",home_loc_all)        
 
'''
"""
def Target_payload():
        global check_box_flag3       
        global xy_pos, latlon_pos
        global master, no_uavs
        global self_heal
        global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10, vehicle11, vehicle12, vehicle13, vehicle14,vehicle15,vehicle16,vehicle17,vehicle18,vehicle19,vehicle20,vehicle21,vehicle22,vehicle23,vehicle24,vehicle25
        global follower_host_tuple
        global circle_pos_flag

        xoffset = xoffset_entry.get()
        cradius = cradius_entry.get()
        aoffset = aoffset_entry.get()
        salt = salt_entry.get() 
        xoffset = int(xoffset)
        cradius = int(cradius)
        aoffset = int(aoffset)
        salt = int(salt)
        print ("aggregation", master)
        
        
        if master == 1:
            lat = vehicle1.location.global_relative_frame.lat
            lon = vehicle1.location.global_relative_frame.lon

        if master == 2:
            lat = vehicle2.location.global_relative_frame.lat
            lon = vehicle2.location.global_relative_frame.lon

        if master == 3:
            lat = vehicle3.location.global_relative_frame.lat
            lon = vehicle3.location.global_relative_frame.lon
                 
        if master == 4:
            lat = vehicle4.location.global_relative_frame.lat
            lon = vehicle4.location.global_relative_frame.lon

        if master == 5:
            lat = vehicle5.location.global_relative_frame.lat
            lon = vehicle5.location.global_relative_frame.lon


        counter_payload = 0
        follower_host_tuple_payload = []

        if checkboxvalue_1.get() == 1:
		counter_payload = counter_payload+1
		follower_host_tuple_payload.append(vehicle1)

        if checkboxvalue_2.get() == 1:
		follower_host_tuple_payload.append(vehicle2)
		counter_payload = counter_payload+1

        if checkboxvalue_3.get() == 1:
		follower_host_tuple_payload.append(vehicle3)
		counter_payload = counter_payload+1

        if checkboxvalue_4.get() == 1:
		follower_host_tuple_payload.append(vehicle4)
		counter_payload = counter_payload+1

        if checkboxvalue_5.get() == 1:
		follower_host_tuple_payload.append(vehicle5)
		counter_payload = counter_payload+1

        if checkboxvalue_6.get() == 1:
		follower_host_tuple_payload.append(vehicle6)
		counter_payload = counter_payload+1

        if checkboxvalue_7.get() == 1:
		follower_host_tuple_payload.append(vehicle7)
		counter_payload = counter_payload+1

        if checkboxvalue_8.get() == 1:
		follower_host_tuple_payload.append(vehicle8)
		counter_payload = counter_payload+1

        if checkboxvalue_9.get() == 1:
		follower_host_tuple_payload.append(vehicle9)
		counter_payload = counter_payload+1

        if checkboxvalue_10.get() == 1:
		follower_host_tuple_payload.append(vehicle10)
		counter_payload = counter_payload+1

        if checkboxvalue_11.get() == 1:
		follower_host_tuple_payload.append(vehicle11)
		counter_payload = counter_payload+1

        if checkboxvalue_12.get() == 1:
		follower_host_tuple_payload.append(vehicle12)
		counter_payload = counter_payload+1

        if checkboxvalue_13.get() == 1:
		follower_host_tuple_payload.append(vehicle13)
		counter_payload = counter_payload+1

        if checkboxvalue_14.get() == 1:
		follower_host_tuple_payload.append(vehicle14)
		counter_payload = counter_payload+1

        if checkboxvalue_15.get() == 1:
		follower_host_tuple_payload.append(vehicle15)
		counter_payload = counter_payload+1

        if checkboxvalue_16.get() == 1:
		follower_host_tuple_payload.append(vehicle16)
		counter_payload = counter_payload+1

        if checkboxvalue_17.get() == 1:
		follower_host_tuple_payload.append(vehicle17)
		counter_payload = counter_payload+1

        if checkboxvalue_18.get() == 1:
		follower_host_tuple_payload.append(vehicle18)
		counter_payload = counter_payload+1

        if checkboxvalue_19.get() == 1:
		follower_host_tuple_payload.append(vehicle19)
		counter_payload = counter_payload+1

        if checkboxvalue_20.get() == 1:
		follower_host_tuple_payload.append(vehicle20)
		counter_payload = counter_payload+1

        if checkboxvalue_21.get() == 1:
		follower_host_tuple_payload.append(vehicle21)
		counter_payload = counter_payload+1

        if checkboxvalue_22.get() == 1:
		follower_host_tuple_payload.append(vehicle22)
		counter_payload = counter_payload+1

        if checkboxvalue_23.get() == 1:
		follower_host_tuple_payload.append(vehicle23)
		counter_payload = counter_payload+1

        if checkboxvalue_24.get() == 1:
		follower_host_tuple_payload.append(vehicle24)
		counter_payload = counter_payload+1

        if checkboxvalue_25.get() == 1:
		follower_host_tuple_payload.append(vehicle25)
		counter_payload = counter_payload+1

	print ("....follower_host_tuple_payload...", follower_host_tuple_payload)

        if circle_pos_flag == True:
		circle_pos_flag = False
		clat = clat_entry.get()
		clon = clon_entry.get()
		clat = float(clat)
		clon = float(clon)
		formation(int(counter_payload), 'C', clat, clon)
        else:
        	formation(int(counter_payload), 'C', lat, lon)

        original_location = [lat, lon]


        a,b,c = (0,0,0)
        count_wp = 0

        print ("................", latlon_pos)
        #for i in range(0, int(no_uavs)):  
        alt_001 = salt
        for i, iter_follower_payload in enumerate(follower_host_tuple_payload):
            if iter_follower_payload == None:
                ##print "lost odd uav", self_heal[i]
                print ("payload drop lost  uav", (i+1)) 
                if check_box_flag3 == True: 
                	print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
                	#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
            else: 
                #if i < int(no_uavs):   
                print ("payload present  uav :", (i+1))    
                ###init_pos1 = altered_position(original_location,a,b)  
                test = latlon_pos[i]
                print ("..t_goto..", test[0], test[1])    
                alt_123 = iter_follower_payload.location.global_relative_frame.alt  
		print ("...............Dhikshith..........", alt_123)
                target = LocationGlobalRelative(test[0], test[1], alt_123)
                print ("target", target)
		for i in range(0, 5):
                	aggregation_formation(iter_follower_payload,"GUIDED", target)        
                #alt_001 = alt_001 + aoffset+10

"""

"""
def Target_payload():
	global master

	export_mission_filename = 'exportedmission.txt'
        
        ##if master == 1:
		##save_mission(vehicle1, export_mission_filename)
        ##elif master == 2:
		##save_mission(vehicle2, export_mission_filename)
        ##elif master == 3:
		##save_mission(vehicle3, export_mission_filename)
	##time.sleep(1)
        
	aFileName = export_mission_filename


	print("\nReading mission from file: %s" % aFileName)
	print ("waypoint_aggregation")
	missionlist=[]


	with open(aFileName) as f:
		for i, line in enumerate(f):  
		    if i==0:
			if not line.startswith('QGC WPL 110'):
			    raise Exception('File is not supported WP version')
		    elif i==1:
			    print ("first way point reject")
		    else:
		    
			    linearray=line.split('\t')
			    ln_index=int(linearray[0])
			    ln_currentwp=int(linearray[1])
			    ln_frame=int(linearray[2])
			    ln_command=int(linearray[3])
			    ln_param1=float(linearray[4])
			    ln_param2=float(linearray[5])
			    ln_param3=float(linearray[6])
			    ln_param4=float(linearray[7])
			    ln_param5=float(linearray[8])
			    ln_param6=float(linearray[9])
			    ln_param7=float(linearray[10])
			    ln_autocontinue=int(linearray[11].strip())
			    cmd = Command( 0, 0, 0, 3, 16, 0, 1, 0.0, 0.0, 0.0, 0.0, ln_param5, ln_param6, ln_param7)
			    missionlist.append(cmd)
	if master == 1:
		cmdline(vehicle1,missionlist)
	elif master == 2:
		cmdline(vehicle2,missionlist)
	elif master == 3:
		cmdline(vehicle3,missionlist)
"""


def print_path():
    f = tkinter.filedialog.askopenfilename(
        parent=root, initialdir='C:/Tutorial',
        title='Choose file',
        filetypes=[('png images', '.png'),
                   ('gif images', '.jpg')]
        )

    print(f)

"""
def CSV_READ():
    global self_heal_manual_flag
    
    # device's IP address
    print ("add file")
    b1 = tkinter.Button(text='Print path', command=print_path)
    b1.pack(fill='x')
    
    print ("....manual self heal....")
    self_heal_manual_flag = True
    aggr()
    self_heal_manual_flag = False
  
    SERVER_HOST = "192.168.6.8"
    SERVER_PORT = 5002
    # receive 4096 bytes each time
    BUFFER_SIZE = 4096
    SEPARATOR = "<SEPARATOR>"
    # create the server socket
    # TCP socket
    s = socket.socket()
    # bind the socket to our local address
    s.bind((SERVER_HOST, SERVER_PORT))
    # enabling our server to accept connections
    # 5 here is the number of unaccepted connections that
    # the system will allow before refusing new connections
    s.listen(5)
    print(f"[*] Listening as {SERVER_HOST}:{SERVER_PORT}")
    # accept connection if there is any
    client_socket, address = s.accept() 
    # if below code is executed, that means the sender is connected
    print(f"[+] {address} is connected.")
    # receive the file infos
    # receive using client socket, not server socket
    received = client_socket.recv(BUFFER_SIZE).decode()
    filename, filesize = received.split(SEPARATOR)
    # remove absolute path if there is
    filename = os.path.basename(filename)
    # convert to integer
    filesize = int(filesize)
    # start receiving the file from the socket
    # and writing to the file stream
    progress = tqdm.tqdm(range(filesize), f"Receiving {filename}", unit="B", unit_scale=True, unit_divisor=1024)
    with open(filename, "wb") as f:
        for _ in progress:
            # read 1024 bytes from the socket (receive)
            bytes_read = client_socket.recv(BUFFER_SIZE)
            if not bytes_read:    
                # nothing is received
                # file transmitting is done
                break
            # write to the file the bytes we just received
            f.write(bytes_read)
            # update the progress bar
            progress.update(len(bytes_read))

    # close the client socket
    client_socket.close()
    # close the server socket
    s.close()
    """

def air_break(vehicle):
    if vehicle.armed:
        print('\n')
        print('{} - Calling function air_break().'.format(time.ctime()))
        
        msg = vehicle.message_factory.set_position_target_local_ned_encode(
            0,       # time_boot_ms (not used)
            0, 0,    # target system, target component
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, # frame
            0b0000111111000111, # type_mask (only speeds enabled)
            0, 0, 0, # x, y, z positions (not used)
            0, 0, 0, # x, y, z velocity in m/s
            0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
            0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)

        # Send message one time, then check the speed, if not stop, send again.
        print('{} - Sending air break command first time.'.format(time.ctime()))
        vehicle.send_mavlink(msg)
        #get_vehicle_state(vehicle)
        while ((vehicle.velocity[0]**2+vehicle.velocity[1]**2+vehicle.velocity[2]**2)>0.1):
            print('{} - Sending air break command once again.'.format(time.ctime()))
            vehicle.send_mavlink(msg)
            print('{} - Body Frame Velocity command is sent! Vx={}, Vy={}, Vz={}'.format(time.ctime(), vehicle.velocity[0], vehicle.velocity[1], vehicle.velocity[2]))
            time.sleep(1)
            #get_vehicle_state(vehicle)
            print('\n')
    else:
        print('{} - Vehicle is not armed, no need to break.'.format(time.ctime()))

def distance_guided(tlat__, tlon__, talt__):
	global self_heal, follower_host_tuple, master, wp_navigation_stop_forward_flag, wp_navigation_stop_return_flag
	global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10, vehicle11, vehicle12, vehicle13, vehicle14,vehicle15,vehicle16,vehicle17,vehicle18,vehicle19,vehicle20,vehicle21,vehicle22,vehicle23,vehicle24,vehicle25
	global goto_lat_g,goto_lon_g,goto_alt_g
	global self_heal_move_flag
	R = 6373.0
	print ("..hi...tlat, tlon ,talt", tlat__, tlon__, talt__)
	while True:
		try:	
	    		if master == 1:	
				lat__1 = radians(vehicle1.location.global_frame.lat)
				lon__1 = radians(vehicle1.location.global_frame.lon)

	    		if master == 2:
				lat__1 = radians(vehicle2.location.global_frame.lat)
				lon__1 = radians(vehicle2.location.global_frame.lon)

	    		if master == 3:
				lat__1 = radians(vehicle3.location.global_frame.lat)
				lon__1 = radians(vehicle3.location.global_frame.lon)

	    		if master == 4:
				lat__1 = radians(vehicle4.location.global_frame.lat)
				lon__1 = radians(vehicle4.location.global_frame.lon)

	    		if master == 5:
				lat__1 = radians(vehicle5.location.global_frame.lat)
				lon__1 = radians(vehicle5.location.global_frame.lon)

	
			lat__2 = radians(tlat__)
			lon__2 = radians(tlon__)
			dlon__ = lon__2 - lon__1
			dlat__ = lat__2 - lat__1
			a__ = sin(dlat__ / 2)**2 + cos(lat__1) * cos(lat__2) * sin(dlon__ / 2)**2
			c__ = 2 * atan2(sqrt(a__), sqrt(1 - a__))
			distance__1 = R * c__
			distance_x = int(distance__1 * 1000)
			print ("....master.....", master)
			print ("....>>>...tlat, tlon", tlat__, tlon__)
			print(">>>>>............distance1:", distance_x)
			print (",,,self_heal_move_flag..", self_heal_move_flag)

		except:	
			pass
	
		if self_heal_move_flag == True:
			print (".....self_heal_move_flag == True.....")
			print ("one uav rtl or lost")
			print ("...self_heal..", self_heal)
			for i, iter_follower in enumerate(follower_host_tuple): 
				if self_heal[i] > 0:
					print ("lost odd uav", (i+1))
				else:
					print ("present uav")  
					for i in range(0, 5):
						if iter_follower.mode.name =="RTL":
							dataf = 10
						else:
							threading.Thread(target=air_break,args=(iter_follower,)).start()
							#air_break(iter_follower)
							time.sleep(0.2)
                        #..............RTL...............
			time.sleep(1)
			for i in range(0, 2):
				altitude()
			time.sleep(1)
			aggr()
			time.sleep(1)
                        #..................................
			move_all_pos(float(tlat__),float(tlon__),int(talt__), 'forward')
			self_heal_move_flag = False
			#break
		if wp_navigation_stop_forward_flag == True:
		    	print ("......break......wp_navigation_stop_forward_flag")
		    	break
		if wp_navigation_stop_return_flag == True:
		    	print ("......break......wp_navigation_stop_return_flag")
			break
		try:
			if distance_x < 4:
				print ("reached location")
				break
		except:
			pass
		time.sleep(0.2)


def move_all_pos_guided():   #move_all
    global wp_navigation_guided_forward_flag
    global wp_navigation_guided_return_flag
    global wp_navigation_flag, aggr_and_rtl_flag, wp_navigation_stop_forward_flag
    global master,wp_pos
    global goto_lat_g,goto_lon_g,goto_alt_g
    global self_heal_odd, self_heal_even
    global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10
    global follower_host_tuple_pos, follower_host_tuple_neg
    global RTH_flag
    global control_command, count_123
    global no_uavs

    print ("...wp_pos.....",wp_pos)
    while True:
	    time.sleep(1)
            if control_command == True or wp_navigation_guided_forward_flag == True:
		    try:
			if wp_navigation_flag == True or wp_navigation_guided_forward_flag == True:
			    #wp_navigation_flag = False
			    send_01()

			
			    for i in range(0, len(wp_pos)):
				    print ("wp_navigation_flag.........enable...")
				    goto_lat_g=wp_pos[i][0]
				    goto_lon_g=wp_pos[i][1]
				    goto_alt_g=wp_pos[i][2]
				    print ("....goto_lat_g,goto_lon_g,goto_alt_g..", goto_lat_g,goto_lon_g,goto_alt_g)
                                    if i == len(wp_pos)-1:
					print ("last forward point")
					move_all_pos_last(float(goto_lat_g),float(goto_lon_g),int(goto_alt_g), 'forward')
				    else:
				    	move_all_pos(float(goto_lat_g),float(goto_lon_g),int(goto_alt_g), 'forward')
				    print (".?...wp1..")
				    distance_guided(float(goto_lat_g),float(goto_lon_g), int(goto_alt_g))
				    #aggr()
				    time.sleep(4)
				    if wp_navigation_guided_forward_flag == False:
					    if i == 0:
						print ("initial position flocking")   #........................flock position....time 30 sec
						time.sleep(0.5)
                                    """
				    if RTH_flag == True:
					    time.sleep(10)
					    altitude()
					    time.sleep(15)
					    aggr()
					    time.sleep(10)
					    RTH_flag = False
                                    """
				    print (".......?????....muthu...")
				    if wp_navigation_stop_forward_flag == True:
			    		print ("......break......wp_navigation_stop_forward_flag")
					break
                            """
			    if master == 1:
				formation_heading(no_uavs, 'L', vehicle1.location.global_frame.lat, vehicle1.location.global_frame.lon, 270)
			    if master == 2:
				formation_heading(no_uavs, 'L', vehicle2.location.global_frame.lat, vehicle2.location.global_frame.lon, 270)
			    if master == 3:
				formation_heading(no_uavs, 'L', vehicle3.location.global_frame.lat, vehicle3.location.global_frame.lon, 270)
			    if master == 4:
				formation_heading(no_uavs, 'L', vehicle4.location.global_frame.lat, vehicle4.location.global_frame.lon, 270)
                            time.sleep(2)
			    aggr_line()
                            """
		            wp_navigation_flag = False
			    if wp_navigation_guided_forward_flag == True:
				print ("start forward mission only")
				wp_navigation_guided_forward_flag = False
			    else:
				    print ("wp upload to all uav start")
				    aggr_and_rtl_flag = True
				    time.sleep(1)
				    aggr()
				    generate_search_misison()
				    time.sleep(1)
				    for i in range(0,5):
				    	auto()
					
			    wp_navigation_guided_forward_flag = False
			    wp_navigation_stop_forward_flag = False
			time.sleep(1)
		    except:
			    time.sleep(1)
			    pass

def move_all_pos_guided_return():   #move_all
    global wp_navigation_guided_forward_flag
    global wp_navigation_guided_return_flag
    global wp_navigation_return_flag, aggr_and_rtl_flag, wp_navigation_stop_return_flag
    global master, wp_pos, control_command
    global self_heal_odd, self_heal_even
    global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10
    global follower_host_tuple_pos, follower_host_tuple_neg
    global goto_lat_g,goto_lon_g,goto_alt_g
    global control_command, count_123

    export_mission_filename = 'exportedmission_01.txt'
    aFileName = export_mission_filename
    print("\nReading mission from file: %s" % aFileName)
    print ("waypoint_aggregation")

    print ("...wp_pos.....",wp_pos)
    while True:
	    #try:
        time.sleep(1)
        if control_command == True or wp_navigation_guided_return_flag == True:
		if wp_navigation_return_flag == True or wp_navigation_guided_return_flag == True:
		    print ("wp_navigation_return flag.........enable...")
		    #wp_navigation_return_flag = False
		    rece_01()
		    wp_navigation_guided_return_flag = False
		    length_pos = len(wp_pos)
		    print ("...length_pos...", length_pos)
		    print ("...", wp_pos)
		    for i in range(0, length_pos):
			    print ("wp_navigation_return.........enable...", length_pos-i)
			    goto_lat_g=wp_pos[length_pos-1-i][0]
			    goto_lon_g=wp_pos[length_pos-1-i][1]
			    goto_alt_g=wp_pos[length_pos-1-i][2]
			    print ("....goto_lat_g,goto_lon_g,goto_alt_g..", goto_lat_g,goto_lon_g,goto_alt_g)
			    move_all_pos(float(goto_lat_g),float(goto_lon_g),int(goto_alt_g), 'return')
			    print ("....wp1..")
			    distance_guided(float(goto_lat_g),float(goto_lon_g),int(goto_alt_g))
			    #aggr()
			    #time.sleep(5)

			    if wp_navigation_stop_return_flag == True:

			    	print ("......break......wp_navigation_stop_return_flag")
				break

		    print ("reach home point aggr and rtl")
		    wp_navigation_return_flag = False
		    wp_navigation_stop_return_flag = False
		    print ("home position move")
                    """
		    for i in range(0,2):
		    	Home_pos()
		    	time.sleep(2)
                    time.sleep(30)
	            print ("mode RTH all UAV")
		    for i in range(0,2):
		    	rtl()	
		    	time.sleep(2)
                    """

"""
def move_all_pos_guided_return():   #move_all
    global wp_navigation_return_flag, aggr_and_rtl_flag
    global master
    global self_heal_odd, self_heal_even
    global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10
    global follower_host_tuple_pos, follower_host_tuple_neg
    while True:
	    try:
		if wp_navigation_return_flag == True:
		    print ("wp_navigation_return flag.........enable...")
		    wp_navigation_return_flag = False
		    goto_lat=g_lat5.get()
		    goto_lon=g_lon5.get()
		    print ("..1..goto_lat,goto_lon..", goto_lat,goto_lon)
		    move_all_pos(float(goto_lat),float(goto_lon))
		    print ("....wp1..")
		    distance_guided(float(goto_lat),float(goto_lon))
		    print ("....distance_guided..1..")

		    goto_lat=g_lat6.get()
		    goto_lon=g_lon6.get()
		    print ("..2..goto_lat,goto_lon..", goto_lat,goto_lon)
		    move_all_pos(float(goto_lat),float(goto_lon))
		    distance_guided(float(goto_lat),float(goto_lon))

		    goto_lat=g_lat7.get()
		    goto_lon=g_lon7.get()
		    print ("..3..goto_lat,goto_lon..", goto_lat,goto_lon)
		    move_all_pos(float(goto_lat),float(goto_lon))
		    distance_guided(float(goto_lat),float(goto_lon))
		    print ("reach home point aggr and rtl")
		    aggr()
		    time.sleep(10)  #.....aggr to all uav
		    for i in range(0, 5):
			print ("rtl ...all uav")
		    	rtl()
		time.sleep(1)
	    except:
		    time.sleep(1)
		    pass
"""

def move_all_pos_last(lat, lon, alt, cmd_dir):
    global self_heal  #move_all
    global master, no_uavs
    global xy_pos, latlon_pos
    global self_heal, self_heal
    global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10, vehicle11, vehicle12, vehicle13, vehicle14,vehicle15,vehicle16,vehicle17,vehicle18,vehicle19,vehicle20,vehicle21,vehicle22,vehicle23,vehicle24,vehicle25
    global follower_host_tuple
    global circle_pos_flag
    xoffset = xoffset_entry.get()    
    cradius = cradius_entry.get()    
    aoffset = aoffset_entry.get()    
    salt = salt_entry.get() 
    xoffset = int(xoffset)    
    cradius = int(cradius)    
    aoffset = int(aoffset)
    ##..salt = int(salt)
    salt = int(alt)
    #print ("aggregation", master)

    #print ("...lat, lon...", lat, lon)
    original_location = [lat, lon]

    formation_heading(no_uavs, 'L', lat, lon, 270)
    #print ("....muthuselvam....", original_location)

    a,b,c = (0,0,0)
    count_wp = 0
    alt_000 = salt
    print ("...self_heal..,alt_000", self_heal,alt_000)
    for i, iter_follower in enumerate(follower_host_tuple): 
        #i = i+1
        if self_heal[i] > 0:
            print ("lost odd uav", (i+1))
            pos_latlon = (0.0, 0.0)
            latlon_pos.insert(i, (0.0, 0.0))
            if check_box_flag3 == True: 
		print ("self heal..to alt change")   
            else:
		alt_000 = alt_000 + aoffset 
        else:   
            test = latlon_pos[i]
            print ("..t_goto..", test[0], test[1])   
            if cmd_dir == 'forward':     
            	target = LocationGlobalRelative(test[0], test[1], alt_000)
            if cmd_dir == 'return':     
            	target = LocationGlobalRelative(test[0], test[1], iter_follower.location.global_relative_frame.alt)
	    
            print ("target", target)
	    for i in range(0, 5):
		if iter_follower.mode.name =="RTL":
			dataf = 10
		else:
            		aggregation_formation(iter_follower,"GUIDED", target)        
            alt_000 = alt_000 + aoffset
	    print (".....alt_000...", alt_000)

def move_all_pos(lat, lon, alt, cmd_dir):
    global self_heal  #move_all
    global master, no_uavs
    global xy_pos, latlon_pos
    global self_heal, self_heal
    global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10, vehicle11, vehicle12, vehicle13, vehicle14,vehicle15,vehicle16,vehicle17,vehicle18,vehicle19,vehicle20,vehicle21,vehicle22,vehicle23,vehicle24,vehicle25
    global follower_host_tuple
    global circle_pos_flag
    xoffset = xoffset_entry.get()    
    cradius = cradius_entry.get()    
    aoffset = aoffset_entry.get()    
    salt = salt_entry.get() 
    xoffset = int(xoffset)    
    cradius = int(cradius)    
    aoffset = int(aoffset)
    ##..salt = int(salt)
    salt = int(alt)
    #print ("aggregation", master)

    #print ("...lat, lon...", lat, lon)
    original_location = [lat, lon]

    if checkboxvalue1.get() == 1:
        formation(int(no_uavs), 'T', lat, lon)
    elif checkboxvalue2.get() == 1:
        formation(int(no_uavs), 'L', lat, lon)
    elif checkboxvalue3.get() == 1:
        formation(int(no_uavs), 'S', lat, lon)
    elif checkboxvalue4.get() == 1:
        if circle_pos_flag == True:
            circle_pos_flag = False
            clat = clat_entry.get()
            clon = clon_entry.get()
            clat = float(clat)
            clon = float(clon)
            formation(int(no_uavs), 'C', clat, clon)
        else:
            formation(int(no_uavs), 'C', lat, lon)
    original_location = [lat, lon]

    #print ("....muthuselvam....", original_location)

    a,b,c = (0,0,0)
    count_wp = 0
    alt_000 = salt
    print ("...self_heal..,alt_000", self_heal,alt_000)
    for i, iter_follower in enumerate(follower_host_tuple): 
        #i = i+1
        if self_heal[i] > 0:
            print ("lost odd uav", (i+1))
            pos_latlon = (0.0, 0.0)
            latlon_pos.insert(i, (0.0, 0.0))
            if check_box_flag3 == True: 
		print ("self heal..to alt change")   
            else:
		alt_000 = alt_000 + aoffset 
        else:   
            test = latlon_pos[i]
            print ("..t_goto..", test[0], test[1])   
            if cmd_dir == 'forward':     
            	target = LocationGlobalRelative(test[0], test[1], alt_000)
            if cmd_dir == 'return':     
            	target = LocationGlobalRelative(test[0], test[1], iter_follower.location.global_relative_frame.alt)
	    
            print ("target", target)
	    for i in range(0, 5):
		if iter_follower.mode.name =="RTL":
			dataf = 10
		else:
            		aggregation_formation(iter_follower,"GUIDED", target)        
            alt_000 = alt_000 + aoffset
	    print (".....alt_000...", alt_000)



def formation_move(p_lat, p_lon, p_alt, heading_degrees):
    global self_heal  #move_all
    global master, no_uavs
    global xy_pos, latlon_pos
    global self_heal, self_heal
    global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10,vehicle11,vehicle12,vehicle13
    global vehicle14,vehicle15,vehicle16,vehicle17,vehicle18,vehicle19,vehicle20,vehicle21,vehicle22,vehicle23,vehicle24,vehicle25      
    global follower_host_tuple_G1,follower_host_tuple_G2,follower_host_tuple_G3,follower_host_tuple_G4,follower_host_tuple_G5
    global counter_G1,counter_G2,counter_G3,counter_G4,counter_G5
    global follower_host_tuple
    global circle_pos_flag
    xoffset = xoffset_entry.get()    
    cradius = cradius_entry.get()    
    aoffset = aoffset_entry.get()    
    salt = salt_entry.get() 
    xoffset = int(xoffset)    
    cradius = int(cradius)    
    aoffset = int(aoffset)
    salt = int(salt)
    print ("aggregation", master)
    #..................all...................
    if checkboxvalue_Group_all.get() == 1:
	    formation_heading(no_uavs, 'L', p_lat, p_lon, heading_degrees)
            time.sleep(1)
	    a,b,c = (0,0,0)
	    count_wp = 0
	    alt_000 = salt

	    for i, iter_follower in enumerate(follower_host_tuple): 
		#i = i+1
		if self_heal[i] > 0:
		    ##print "lost odd uav", self_heal[i]
		    print ("lost odd uav", (i+1))
		    pos_latlon = (0.0, 0.0)
		    latlon_pos.insert(i, (0.0, 0.0))
		    ##c = (c+20)   
		    if check_box_flag3 == True: 
		        print ("self heal..to alt change")   
		    else:
		        alt_000 = alt_000 + aoffset 
		else:   
		    test = latlon_pos[i]
		    print ("..t_goto..", test[0], test[1])        
		    target = LocationGlobalRelative(test[0], test[1], alt_000)
		    print ("target", target)
		    for i in range(0, 5):
		    	aggregation_formation(iter_follower,"GUIDED", target)        
		    alt_000 = alt_000 + aoffset
	    #...................end..................


def search(): 

    global self_heal  #move_all
    global master, no_uavs
    global xy_pos, latlon_pos
    global self_heal, self_heal
    global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10,vehicle11,vehicle12,vehicle13
    global vehicle14,vehicle15,vehicle16,vehicle17,vehicle18,vehicle19,vehicle20,vehicle21,vehicle22,vehicle23,vehicle24,vehicle25      
    global follower_host_tuple_G1,follower_host_tuple_G2,follower_host_tuple_G3,follower_host_tuple_G4,follower_host_tuple_G5
    global counter_G1,counter_G2,counter_G3,counter_G4,counter_G5
    global follower_host_tuple
    global circle_pos_flag
    xoffset = xoffset_entry.get()    
    cradius = cradius_entry.get()    
    aoffset = aoffset_entry.get()    
    salt = salt_entry.get() 
    xoffset = int(xoffset)    
    cradius = int(cradius)    
    aoffset = int(aoffset)
    salt = int(salt)
    print ("aggregation", master)
    #...................group1..................
    if checkboxvalue_Group_1.get() == 1:
        Group_1()
        try:
		G1_M = G1_master_set_entry.get()
		G1_M = int(G1_M)
		if G1_M == 1:
		    goto_lat=g_lat1.get()
		    goto_lon=g_lon1.get()

		    lat=float(goto_lat)
		    lon=float(goto_lon)
		    alt_0 = vehicle1.location.global_relative_frame.alt

		if G1_M == 2:
		    goto_lat=g_lat2.get()
		    goto_lon=g_lon2.get()

		    lat=float(goto_lat)
		    lon=float(goto_lon)
		    alt_0 = vehicle2.location.global_relative_frame.alt

		if G1_M == 3:
		    goto_lat=g_lat3.get()
		    goto_lon=g_lon3.get()

		    lat=float(goto_lat)
		    lon=float(goto_lon)
		    alt_0 = vehicle3.location.global_relative_frame.alt
			 


		if checkboxvalue1.get() == 1:
		    formation(counter_G1, 'T', lat, lon)
		elif checkboxvalue2.get() == 1:
		    formation(counter_G1, 'L', lat, lon)
		elif checkboxvalue3.get() == 1:
		    formation(counter_G1, 'S', lat, lon)
		elif checkboxvalue4.get() == 1:
		    if circle_pos_flag == True:
			circle_pos_flag = False
			clat = clat_entry.get()
			clon = clon_entry.get()
			clat = float(clat)
			clon = float(clon)
			formation(counter_G1, 'C', clat, clon)
		    else:
			formation(counter_G1, 'C', lat, lon)

		print ("................", latlon_pos)
		#for i in range(0, int(no_uavs)):  
		alt_001 = salt
		for i, iter_follower_G1 in enumerate(follower_host_tuple_G1):
		    if iter_follower_G1 == None:
			##print "lost odd uav", self_heal[i]
			print ("payload drop lost  uav", (i+1)) 
			if check_box_flag3 == True: 
				print ("self heal..to alt change")   
			else:
				alt_001 = alt_001
				#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
		    else: 
			#if i < int(no_uavs):   
			print ("payload present  uav :", (i+1))    
			###init_pos1 = altered_position(original_location,a,b)  
			test = latlon_pos[i]
			print ("..t_goto..", test[0], test[1])    
			alt_123 = iter_follower_G1.location.global_relative_frame.alt  
			print ("...............Dhikshith..........", alt_123)
			target = LocationGlobalRelative(test[0], test[1], alt_123)
			print ("target", target)
			for i in range(0, 5):
				aggregation_formation(iter_follower_G1,"GUIDED", target) 
        except:
		pass  
    #...................group2..................

    if checkboxvalue_Group_2.get() == 1:
        Group_2()
        try:
		G2_M = G2_master_set_entry.get()
		G2_M = int(G2_M)

		if G2_M == 4:
		    goto_lat=g_lat4.get()
		    goto_lon=g_lon4.get()

		    lat=float(goto_lat)
		    lon=float(goto_lon)
		    alt_0 = vehicle4.location.global_relative_frame.alt

		if G2_M == 5:
		    goto_lat=g_lat5.get()
		    goto_lon=g_lon5.get()

		    lat=float(goto_lat)
		    lon=float(goto_lon)
		    alt_0 = vehicle5.location.global_relative_frame.alt

		if G2_M == 6:
		    goto_lat=g_lat6.get()
		    goto_lon=g_lon6.get()

		    lat=float(goto_lat)
		    lon=float(goto_lon)
		    alt_0 = vehicle6.location.global_relative_frame.alt



		if checkboxvalue1.get() == 1:
		    formation(counter_G2, 'T', lat, lon)
		elif checkboxvalue2.get() == 1:
		    formation(counter_G2, 'L', lat, lon)
		elif checkboxvalue3.get() == 1:
		    formation(counter_G2, 'S', lat, lon)
		elif checkboxvalue4.get() == 1:
		    if circle_pos_flag == True:
			circle_pos_flag = False
			clat = clat_entry.get()
			clon = clon_entry.get()
			clat = float(clat)
			clon = float(clon)
			formation(counter_G2, 'C', clat, clon)
		    else:
			formation(counter_G2, 'C', lat, lon)

		print ("follower_host_tuple_G2", follower_host_tuple_G2)
		print ("counter_G2", counter_G2)
		print ("................", latlon_pos)
		#for i in range(0, int(no_uavs)):  
		alt_001 = salt
		for i, iter_follower_G2 in enumerate(follower_host_tuple_G2):
		    if iter_follower_G2 == None:
			##print "lost odd uav", self_heal[i]
			print ("payload drop lost  uav", (i+1)) 
			if check_box_flag3 == True: 
				print ("self heal..to alt change")   
			else:
				alt_001 = alt_001
				#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
		    else: 
			#if i < int(no_uavs):   
			print ("payload present  uav :", (i+1))    
			###init_pos1 = altered_position(original_location,a,b)  
			test = latlon_pos[i]
			print ("..t_goto..", test[0], test[1])    
			alt_123 = iter_follower_G2.location.global_relative_frame.alt  
			print ("...............Dhikshith..........", alt_123)
			target = LocationGlobalRelative(test[0], test[1], alt_123)
			print ("target", target)
			for i in range(0, 5):
				aggregation_formation(iter_follower_G2,"GUIDED", target)  
        except:
		pass  
    #...................group3..................

    if checkboxvalue_Group_3.get() == 1:
        Group_3()
        try:
		G3_M = G3_master_set_entry.get()
		G3_M = int(G3_M)
		if G3_M == 11:
		    goto_lat=g_lat11.get()
		    goto_lon=g_lon11.get()

		    lat=float(goto_lat)
		    lon=float(goto_lon)
		    alt_0 = vehicle1.location.global_relative_frame.alt

		if G3_M == 12:
		    goto_lat=g_lat12.get()
		    goto_lon=g_lon12.get()

		    lat=float(goto_lat)
		    lon=float(goto_lon)
		    alt_0 = vehicle12.location.global_relative_frame.alt

		if G3_M == 13:
		    goto_lat=g_lat13.get()
		    goto_lon=g_lon13.get()

		    lat=float(goto_lat)
		    lon=float(goto_lon)
		    alt_0 = vehicle13.location.global_relative_frame.alt
			 
		if G3_M == 14:
		    goto_lat=g_lat14.get()
		    goto_lon=g_lon14.get()

		    lat=float(goto_lat)
		    lon=float(goto_lon)
		    alt_0 = vehicle14.location.global_relative_frame.alt

		if G3_M == 15:
		    goto_lat=g_lat15.get()
		    goto_lon=g_lon15.get()

		    lat=float(goto_lat)
		    lon=float(goto_lon)
		    alt_0 = vehicle15.location.global_relative_frame.alt
		
		if checkboxvalue1.get() == 1:
		    formation(counter_G3, 'T', lat, lon)
		elif checkboxvalue2.get() == 1:
		    formation(counter_G3, 'L', lat, lon)
		elif checkboxvalue3.get() == 1:
		    formation(counter_G3, 'S', lat, lon)
		elif checkboxvalue4.get() == 1:
		    if circle_pos_flag == True:
			circle_pos_flag = False
			clat = clat_entry.get()
			clon = clon_entry.get()
			clat = float(clat)
			clon = float(clon)
			formation(counter_G3, 'C', clat, clon)
		    else:
			formation(counter_G3, 'C', lat, lon)

		print ("follower_host_tuple_G3", follower_host_tuple_G3)
		print ("counter_G3", counter_G3)
		print ("................", latlon_pos)
		#for i in range(0, int(no_uavs)):  
		alt_001 = salt
		for i, iter_follower_G3 in enumerate(follower_host_tuple_G3):
		    if iter_follower_G3 == None:
			##print "lost odd uav", self_heal[i]
			print ("payload drop lost  uav", (i+1)) 
			if check_box_flag3 == True: 
				print ("self heal..to alt change")   
			else:
				alt_001 = alt_001
				#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
		    else: 
			#if i < int(no_uavs):   
			print ("payload present  uav :", (i+1))    
			###init_pos1 = altered_position(original_location,a,b)  
			test = latlon_pos[i]
			print ("..t_goto..", test[0], test[1])    
			alt_123 = iter_follower_G3.location.global_relative_frame.alt  
			print ("...............Dhikshith..........", alt_123)
			target = LocationGlobalRelative(test[0], test[1], alt_123)
			print ("target", target)
			for i in range(0, 5):
				aggregation_formation(iter_follower_G3,"GUIDED", target)   
        except:
		pass 

    #...................group4..................

    if checkboxvalue_Group_4.get() == 1:
        Group_4()
        try:
		G1_M = G1_master_set_entry.get()
		G1_M = int(G1_M)

		if checkboxvalue1.get() == 1:
		    formation(counter_G4, 'T', lat, lon)
		elif checkboxvalue2.get() == 1:
		    formation(counter_G4, 'L', lat, lon)
		elif checkboxvalue3.get() == 1:
		    formation(counter_G4, 'S', lat, lon)
		elif checkboxvalue4.get() == 1:
		    if circle_pos_flag == True:
			circle_pos_flag = False
			clat = clat_entry.get()
			clon = clon_entry.get()
			clat = float(clat)
			clon = float(clon)
			formation(counter_G4, 'C', clat, clon)
		    else:
			formation(counter_G4, 'C', lat, lon)

		print ("follower_host_tuple_G4", follower_host_tuple_G4)
		print ("counter_G4", counter_G4)

		print ("................", latlon_pos)
		#for i in range(0, int(no_uavs)):  
		alt_001 = salt
		for i, iter_follower_G4 in enumerate(follower_host_tuple_G4):
		    if iter_follower_G4 == None:
			##print "lost odd uav", self_heal[i]
			print ("payload drop lost  uav", (i+1)) 
			if check_box_flag3 == True: 
				print ("self heal..to alt change")   
			else:
				alt_001 = alt_001
				#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
		    else: 
			#if i < int(no_uavs):   
			print ("payload present  uav :", (i+1))    
			###init_pos1 = altered_position(original_location,a,b)  
			test = latlon_pos[i]
			print ("..t_goto..", test[0], test[1])    
			alt_123 = iter_follower_G4.location.global_relative_frame.alt  
			print ("...............Dhikshith..........", alt_123)
			target = LocationGlobalRelative(test[0], test[1], alt_123)
			print ("target", target)
			for i in range(0, 5):
				aggregation_formation(iter_follower_G4,"GUIDED", target)   
        except:
		pass 


    #..................all...................
    if checkboxvalue_Group_all.get() == 1:

	    if master == 1:
		#lat = vehicle1.location.global_relative_frame.lat
		#lon = vehicle1.location.global_relative_frame.lon
		goto_lat=g_lat1.get()
		goto_lon=g_lon1.get()

		lat=float(goto_lat)
		lon=float(goto_lon)
	    if master == 2:
		#lon = vehicle2.location.global_relative_frame.lon
		##alt = vehicle2.location.global_relative_frame.alt
		#alt = 110

		goto_lat=g_lat2.get()
		goto_lon=g_lon2.get()

		lat=float(goto_lat)
		lon=float(goto_lon)

	    if master == 3:
		#lat = vehicle3.location.global_relative_frame.lat
		#lon = vehicle3.location.global_relative_frame.lon
		#alt = 120

		goto_lat=g_lat3.get()
		goto_lon=g_lon3.get()

		lat=float(goto_lat)
		lon=float(goto_lon)

	    if master == 4:
		#lat = vehicle4.location.global_relative_frame.lat
		#lon = vehicle4.location.global_relative_frame.lon
		#alt = 130

		goto_lat=g_lat4.get()
		goto_lon=g_lon4.get()

		lat=float(goto_lat)
		lon=float(goto_lon)

	    if master == 5:
		#lat = vehicle5.location.global_relative_frame.lat
		#lon = vehicle5.location.global_relative_frame.lon
		#alt = 140

		goto_lat=g_lat5.get()
		goto_lon=g_lon5.get()

		lat=float(goto_lat)
		lon=float(goto_lon)

	    if master == 6:
		#lat = vehicle6.location.global_relative_frame.lat
		#lon = vehicle6.location.global_relative_frame.lon
		#alt = 150

		goto_lat=g_lat6.get()
		goto_lon=g_lon6.get()

		lat=float(goto_lat)
		lon=float(goto_lon)

	    if master == 7:
		#lat = vehicle7.location.global_relative_frame.lat
		#lon = vehicle7.location.global_relative_frame.lon
		#alt = 160

		goto_lat=g_lat7.get()
		goto_lon=g_lon7.get()

		lat=float(goto_lat)
		lon=float(goto_lon)

	    if master == 8:
		#lat = vehicle8.location.global_relative_frame.lat
		#lon = vehicle8.location.global_relative_frame.lon
		#alt = 170

		goto_lat=g_lat8.get()
		goto_lon=g_lon8.get()

		lat=float(goto_lat)
		lon=float(goto_lon)

	    if master == 9:
		#lat = vehicle9.location.global_relative_frame.lat
		#lon = vehicle9.location.global_relative_frame.lon
		#alt = 180

		goto_lat=g_lat9.get()
		goto_lon=g_lon9.get()

		lat=float(goto_lat)
		lon=float(goto_lon)

	    if master == 10:
		#lat = vehicle10.location.global_relative_frame.lat
		#lon = vehicle10.location.global_relative_frame.lon
		#alt = 190

		goto_lat=g_lat10.get()
		goto_lon=g_lon10.get()

		lat=float(goto_lat)
		lon=float(goto_lon)
	    


	    if checkboxvalue1.get() == 1:
		formation(int(no_uavs), 'T', lat, lon)
	    elif checkboxvalue2.get() == 1:
		formation(int(no_uavs), 'L', lat, lon)
	    elif checkboxvalue3.get() == 1:
		formation(int(no_uavs), 'S', lat, lon)
	    elif checkboxvalue4.get() == 1:
		if circle_pos_flag == True:
		    circle_pos_flag = False
		    clat = clat_entry.get()
		    clon = clon_entry.get()
		    clat = float(clat)
		    clon = float(clon)
		    formation(int(no_uavs), 'C', clat, clon)
		else:
		    formation(int(no_uavs), 'C', lat, lon)


	    a,b,c = (0,0,0)
	    count_wp = 0
	    alt_000 = salt

	    for i, iter_follower in enumerate(follower_host_tuple): 
		#i = i+1
		if self_heal[i] > 0:
		    ##print "lost odd uav", self_heal[i]
		    print ("lost odd uav", (i+1))
		    pos_latlon = (0.0, 0.0)
		    latlon_pos.insert(i, (0.0, 0.0))
		    ##c = (c+20)   
		    if check_box_flag3 == True: 
		        print ("self heal..to alt change")   
		    else:
		        alt_000 = alt_000 + aoffset 
		else:   
		    test = latlon_pos[i]
		    print ("..t_goto..", test[0], test[1])        
		    target = LocationGlobalRelative(test[0], test[1], alt_000)
		    print ("target", target)
		    aggregation_formation(iter_follower,"GUIDED", target)        
		    alt_000 = alt_000 + aoffset
	    #...................end..................
'''
def altitude():

        global self_heal
        global follower_host_tuple
        count_wp = 0
        altd = altd_entry.get()
        altd = int(altd)
        aoffset = aoffset_entry.get()
        aoffset = int(aoffset)
        global check_box_flag3       
        global xy_pos, latlon_pos
        global master, no_uavs
        global self_heal
        global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10,vehicle11,vehicle12,vehicle13
        global vehicle14,vehicle15,vehicle16,vehicle17,vehicle18,vehicle19,vehicle20,vehicle21,vehicle22,vehicle23,vehicle24,vehicle25      
	global follower_host_tuple_G1,follower_host_tuple_G2,follower_host_tuple_G3,follower_host_tuple_G4,follower_host_tuple_G5
	global counter_G1,counter_G2,counter_G3,counter_G4,counter_G5

	global follower_host_tuple
        global circle_pos_flag

        xoffset = xoffset_entry.get()
        cradius = cradius_entry.get()
        aoffset = aoffset_entry.get()
        salt = salt_entry.get() 
        xoffset = int(xoffset)
        cradius = int(cradius)
        aoffset = int(aoffset)
        salt = int(salt)
        print ("aggregation", master)

	#................group 1......................
        if checkboxvalue_Group_1.get() == 1:
                Group_1()
                try:
 
			alt_001 = salt
			for i, iter_follower_G1 in enumerate(follower_host_tuple_G1):
			    if iter_follower_G1 == None:
				print ("payload drop lost  uav", (i+1)) 
			    else: 
				print ("payload present  uav :", (i+1))    
	
				for i in range(0, 5):
					altitude_inc(iter_follower_G1,altd)
					time.sleep(0.1)  
				altd = altd+aoffset 
                except:
			pass
	#................group 2......................
        if checkboxvalue_Group_2.get() == 1:
                Group_2()
                try:
 
			alt_001 = salt
			for i, iter_follower_G2 in enumerate(follower_host_tuple_G2):
			    if iter_follower_G2 == None:
				print ("payload drop lost  uav", (i+1)) 
			    else: 
				print ("payload present  uav :", (i+1))    
	
				for i in range(0, 5):
					altitude_inc(iter_follower_G2,altd)
					time.sleep(0.1) 
				altd = altd+aoffset  
                except:
			pass
	#................group 3......................
        if checkboxvalue_Group_3.get() == 1:
                Group_3()
                try:
 
			alt_001 = salt
			for i, iter_follower_G3 in enumerate(follower_host_tuple_G3):
			    if iter_follower_G3 == None:
				print ("payload drop lost  uav", (i+1)) 
			    else: 
				print ("payload present  uav :", (i+1))    
	
				for i in range(0, 5):
					altitude_inc(iter_follower_G3,altd) 
					time.sleep(0.1) 
				altd = altd+aoffset 
                except:
			pass
	#................group 4......................
        if checkboxvalue_Group_4.get() == 1:
                Group_4()
                try:
 
			alt_001 = salt
			for i, iter_follower_G4 in enumerate(follower_host_tuple_G4):
			    if iter_follower_G4 == None:
				print ("payload drop lost  uav", (i+1)) 
			    else: 
				print ("payload present  uav :", (i+1))    
	
				for i in range(0, 5):
					altitude_inc(iter_follower_G4,altd)
					time.sleep(0.1) 
				altd = altd+aoffset 
                except:
			pass
	#................group 5......................
	
        if checkboxvalue_Group_all.get() == 1:
		print ("..self_heal...", self_heal)
                try:
			alt_001 = salt
			for i, iter_follower in enumerate(follower_host_tuple):
			    if self_heal[i] > 0:
				print ("payload drop lost  uav", (i+1)) 
				if check_box_flag3 == True: 
		        		print ("self heal..to alt change")   
				else:
		        		altd = altd + aoffset
		    
			    else: 
				print ("payload present  uav :", (i+1))    
	
				for i in range(0, 5):
					altitude_inc(iter_follower,altd) 
					time.sleep(0.1) 
				altd = altd+aoffset 
                except:

			pass

def altitude_same():
        global self_heal
        global vehicle1
        global follower_host_tuple
        alts = alts_entry.get()
        alts = int(alts)
        count_wp = 0
        global check_box_flag3       
        global xy_pos, latlon_pos
        global master, no_uavs
        global self_heal
        global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10,vehicle11,vehicle12,vehicle13
        global vehicle14,vehicle15,vehicle16,vehicle17,vehicle18,vehicle19,vehicle20,vehicle21,vehicle22,vehicle23,vehicle24,vehicle25      
	global follower_host_tuple_G1,follower_host_tuple_G2,follower_host_tuple_G3,follower_host_tuple_G4,follower_host_tuple_G5
	global counter_G1,counter_G2,counter_G3,counter_G4,counter_G5

	global follower_host_tuple
        global circle_pos_flag

        xoffset = xoffset_entry.get()
        cradius = cradius_entry.get()
        aoffset = aoffset_entry.get()
        salt = salt_entry.get() 
        xoffset = int(xoffset)
        cradius = int(cradius)
        aoffset = int(aoffset)
        salt = int(salt)
        print ("aggregation", master)
	#................group 1......................
        if checkboxvalue_Group_1.get() == 1:
                Group_1()
                try:
 
			alt_001 = salt
			for i, iter_follower_G1 in enumerate(follower_host_tuple_G1):
			    if iter_follower_G1 == None:
				print ("payload drop lost  uav", (i+1)) 
			    else: 
				print ("payload present  uav :", (i+1))    
	
				for i in range(0, 5):
					altitude_inc(iter_follower_G1,alts) 
					time.sleep(0.1) 
                except:
			pass
	#................group 2......................
        if checkboxvalue_Group_2.get() == 1:
                Group_2()
                try:
 
			alt_001 = salt
			for i, iter_follower_G2 in enumerate(follower_host_tuple_G2):
			    if iter_follower_G2 == None:
				print ("payload drop lost  uav", (i+1)) 
			    else: 
				print ("payload present  uav :", (i+1))    
	
				for i in range(0, 5):
					altitude_inc(iter_follower_G2,alts) 
					time.sleep(0.1) 
                except:
			pass
	#................group 3......................
        if checkboxvalue_Group_3.get() == 1:
                Group_3()
                try:
 
			alt_001 = salt
			for i, iter_follower_G3 in enumerate(follower_host_tuple_G3):
			    if iter_follower_G3 == None:
				print ("payload drop lost  uav", (i+1)) 
			    else: 
				print ("payload present  uav :", (i+1))    
	
				for i in range(0, 5):
					altitude_inc(iter_follower_G3,alts) 
					time.sleep(0.1) 
                except:
			pass
	#................group 4......................
        if checkboxvalue_Group_4.get() == 1:
                Group_4()
                try:
 
			alt_001 = salt
			for i, iter_follower_G4 in enumerate(follower_host_tuple_G4):
			    if iter_follower_G4 == None:
				print ("payload drop lost  uav", (i+1)) 
			    else: 
				print ("payload present  uav :", (i+1))    
	
				for i in range(0, 5):
					altitude_inc(iter_follower_G4,alts) 
					time.sleep(0.1) 
                except:
			pass
	#................group 5......................
        if checkboxvalue_Group_all.get() == 1:

                try:
 
			alt_001 = salt
			for i, iter_follower in enumerate(follower_host_tuple):
			    if self_heal[i] > 0:
				print ("payload drop lost  uav", (i+1)) 
			    else: 
				print ("payload present  uav :", (i+1))    
	
				for i in range(0, 5):
					altitude_inc(iter_follower,alts)
					time.sleep(0.1) 
                except:
			pass
'''



def polar(x,y):

  return math.hypot(x,y),math.degrees(math.atan2(y,x))

def destination_location(homeLattitude, homeLongitude, distance, bearing):
    R = 6371e3 #Radius of earth in metres
    rlat1 = homeLattitude * (math.pi/180) 
    rlon1 = homeLongitude * (math.pi/180)
    d = distance
    bearing = bearing * (math.pi/180) #Converting bearing to radians
    rlat2 = math.asin((math.sin(rlat1) * math.cos(d/R)) + (math.cos(rlat1) * math.sin(d/R) * math.cos(bearing)))
    rlon2 = rlon1 + math.atan2((math.sin(bearing) * math.sin(d/R) * math.cos(rlat1)) , (math.cos(d/R) - (math.sin(rlat1) * math.sin(rlat2))))
    rlat2 = rlat2 * (180/math.pi) #Converting to degrees
    rlon2 = rlon2 * (180/math.pi) #converting to degrees
    location = [rlat2, rlon2]
    return location

def distance_bearing(homeLattitude, homeLongitude, destinationLattitude, destinationLongitude):
    R = 6371e3 #Radius of earth in metres
    rlat1 = homeLattitude * (math.pi/180)
    rlat2 = destinationLattitude * (math.pi/180) 
    rlon1 = homeLongitude * (math.pi/180) 
    rlon2 = destinationLongitude * (math.pi/180) 
    dlat = (destinationLattitude - homeLattitude) * (math.pi/180)
    dlon = (destinationLongitude - homeLongitude) * (math.pi/180)
    #haversine formula to find distance
    a = (math.sin(dlat/2) * math.sin(dlat/2)) + (math.cos(rlat1) * math.cos(rlat2) * (math.sin(dlon/2) * math.sin(dlon/2)))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c #distance in metres
    #formula for bearing
    y = math.sin(rlon2 - rlon1) * math.cos(rlat2)
    x = math.cos(rlat1) * math.sin(rlat2) - math.sin(rlat1) * math.cos(rlat2) * math.cos(rlon2 - rlon1)
    bearing = math.atan2(y, x) #bearing in radians
    bearingDegrees = bearing * (180/math.pi)
    out = [distance, bearingDegrees]
    return out

def geoToCart(origin, endDistance, geoLocation):#origin_lat/lon, 1000, slave_1_lat/lon
    # The initial point of rectangle in (x,y) is (0,0) so considering the current
    # location as origin and retreiving the latitude and longitude from the GPS
    # origin = (12.948048, 80.139742) Format

    # Calculating the hypot end point for interpolating the latitudes and longitudes 
    rEndDistance = math.sqrt(2*(endDistance**2))

    # The bearing for the hypot angle is 45 degrees considering coverage area as square
    bearing = 45

    # Determining the Latitude and Longitude of Middle point of the sqaure area
    # and hypot end point of square area for interpolating latitude and longitude
    lEnd, rEnd = destination_location(origin[0], origin[1], rEndDistance, 180+bearing), destination_location(origin[0], origin[1], rEndDistance, bearing)

    # Array of (x,y)
    x_cart, y_cart  = [-endDistance, 0, endDistance], [-endDistance, 0, endDistance]

    # Array of (latitude, longitude)
    x_lon, y_lat = [lEnd[1], origin[1], rEnd[1]], [lEnd[0], origin[0], rEnd[0]]

    # Latitude interpolation function 
    f_lat = interpolate.interp1d(y_lat, y_cart)

    # Longitude interpolation function
    f_lon = interpolate.interp1d(x_lon, x_cart)

    # Converting (latitude, longitude) to (x,y) using interpolation function
    y, x = f_lat(geoLocation[0]), f_lon(geoLocation[1])

    return (float(x), float(y))

def cartToGeo(origin, endDistance, cartLocation):  #origin_lat/lon, 1000, slave_1_x/y
    # The initial point of rectangle in (x,y) is (0,0) so considering the current
    # location as origin and retreiving the latitude and longitude from the GPS
    # origin = (12.948048, 80.139742) Format

    # Calculating the hypot end point for interpolating the latitudes and longitudes 
    rEndDistance = math.sqrt(2*(endDistance**2))

    # The bearing for the hypot angle is 45 degrees considering coverage area as square
    bearing = 45

    # Determining the Latitude and Longitude of Middle point of the sqaure area
    # and hypot end point of square area for interpolating latitude and longitude
    lEnd, rEnd = destination_location(origin[0], origin[1], rEndDistance, 180+bearing), destination_location(origin[0], origin[1], rEndDistance, bearing)

    # Array of (x,y)
    x_cart, y_cart  = [-endDistance, 0, endDistance], [-endDistance, 0, endDistance]

    # Array of (latitude, longitude)
    x_lon, y_lat = [lEnd[1], origin[1], rEnd[1]], [lEnd[0], origin[0], rEnd[0]]

    # Latitude interpolation function 
    f_lat = interpolate.interp1d(y_cart, y_lat)

    # Longitude interpolation function
    f_lon = interpolate.interp1d(x_cart, x_lon)

    # Converting (latitude, longitude) to (x,y) using interpolation function
    lat, lon = f_lat(cartLocation[0]), f_lon(cartLocation[1])
    return (lat, lon)




def destination_location(homeLattitude, homeLongitude, distance, bearing):
    R = 6371e3 #Radius of earth in metres
    rlat1 = homeLattitude * (math.pi/180) 
    rlon1 = homeLongitude * (math.pi/180)
    d = distance
    bearing = bearing * (math.pi/180) #Converting bearing to radians
    rlat2 = math.asin((math.sin(rlat1) * math.cos(d/R)) + (math.cos(rlat1) * math.sin(d/R) * math.cos(bearing)))
    rlon2 = rlon1 + math.atan2((math.sin(bearing) * math.sin(d/R) * math.cos(rlat1)) , (math.cos(d/R) - (math.sin(rlat1) * math.sin(rlat2))))
    rlat2 = rlat2 * (180/math.pi) #Converting to degrees
    rlon2 = rlon2 * (180/math.pi) #converting to degrees
    location = [rlat2, rlon2]
    return location

def rowsColumns(numDrones):
    perfSquare = math.sqrt(numDrones)
    flag = False

    if ((perfSquare - math.floor(perfSquare)) == 0):
        rows = int(perfSquare)
        columns = int(perfSquare)
        flag = True

    if flag == False:
        combinations = [3, 4, 5, 6, 7, 8, 9, 2]
        for x in combinations:
            if (numDrones % x) == 0:
                columns = x
                rows = int(numDrones/columns)
                flag = True
                break

    return rows, columns, flag

def nodesGeneration_S(numDrones, coverageDistance, lineType):

    #print ("*****nodesGeneration_S", numDrones, coverageDistance, lineType)

    partA, partB, flag = rowsColumns(numDrones)
    scale = coverageDistance
    #incrementalDistance = coverageDistance + (coverageDistance/2)
    length, breadth = ((partA-1) * scale),((partB-1) * scale)
    print(length, breadth)
    coverageDistance, incrementalDistance =  length, breadth
    x, y = np.linspace(0, coverageDistance, partA), np.linspace(0, incrementalDistance, partB)

    xa, xb = np.meshgrid(x, y)


    nodes = np.vstack([xa.ravel(), xb.ravel()]).T
    nrows, ncolumns = xa.shape
    nRowsColumns = [nrows, ncolumns]
    print(incrementalDistance)
    return nodes, coverageDistance

def nodesGeneration_L(numDrones, coverageDistance, lineType):

    flag = True
    #print ("***nodesGeneration_L", numDrones, coverageDistance, lineType)
    coverageDistance = coverageDistance * (numDrones-1)

    # For Horizontal line, linetype is 'H' and for vertical line, linetype is 'V'
    if (lineType != 'H') and (lineType != 'V'):
        x, y = 0, 0
        nodes = [[0,0]]
        flag = False

    else:
        if lineType == 'H':
            x, y = np.linspace(0, coverageDistance, numDrones), np.linspace(0, 0, numDrones)

        if lineType == 'V':
            y, x = np.linspace(0, coverageDistance, numDrones), np.linspace(0, 0, numDrones)

        nodes = np.vstack([x.ravel(), y.ravel()]).T


    return nodes, coverageDistance


def nodesGeneration_T(numDrones, coverageDistance, lineType):
    print ("***nodesGeneration_T", numDrones, coverageDistance, lineType)
    addScale = coverageDistance/20

    # Declaring necessary variables and arrays
    xlist, xval, yval, scale, k = [], [], [], 10, -1

    # Identifying rows count using number of drones
    for i in range(1,int(numDrones/2)):
        eqVal = int(i*(i+1)/2)
        if (eqVal<=numDrones) and ((numDrones - eqVal) > 1):
            rows, eqDrones = i, eqVal

    # Generating the arrays with odd and even numbers
    odd, even = np.array(range(1, 2*rows, 2)), np.array(range(0, 2*rows, 2))

    # Creating the array for triangulation
    for i in range(0, rows):
        xlist.append(even[i:rows-i])
        xlist.append(odd[i:rows-i-1])

    # Creating the array for triangulation
    for xarr in xlist[:rows]:
        for val in xarr:
            xval.append(val)

    # Reversed loop for downward inverted pattern  
    for i in range(rows, 0, -1):  
        # Increment in k after each iteration  
        k += 1  
        for j in range(1, i + 1):  
            yval.append(k)

    # If there's possibility for equilateral triangle
    if eqDrones == numDrones:
        x, y = np.asarray(xval), np.asarray(yval)

    # Generating Equilateral triangle and adding the others in first row
    if eqDrones < numDrones:
        offset = numDrones - eqDrones
        xoff, yoff = np.linspace(0,((rows-1)*2)+2,offset), np.linspace(0,0,offset)
        x, y = np.concatenate([xoff,1+np.asarray(xval)]), np.concatenate([yoff,1+np.asarray(yval)])

    # Scaling the x and y with 10
    ##x, y = 1.5*scale * x, 1.5*scale*y*2
    x, y = addScale*scale * x, addScale*scale*y*2
    """
    # Plotting x and y
    plt.plot(x,y,'ok')
    plt.axis('equal')
    plt.show()
    """
    # Calculating Coverage Distance
    coverageDistance = 1.5*(2 * (rows * scale))

    # Creating x and y as nodes 
    nodes = np.vstack([x.ravel(), y.ravel()]).T

    ##nodes = nodes[::-1]

    return nodes, coverageDistance

def nodesGeneration_C(numDrones, coverageDistance, lineType):

    #print ("***nodesGeneration_C", numDrones, coverageDistance, lineType)

    # Function to Generate Circular Nodes
    # Creating equally spaced 100 data in range 0 to 2*pi
    theta = np.linspace(0, 2 * np.pi, numDrones+1)

    # Setting radius
    radius = coverageDistance

    # Generating x and y data
    x, y = radius * np.cos(theta), radius * np.sin(theta)

    # Omitting Last value because it's repetation of first value
    x, y = x[:-1], y[:-1]
    # print(x,y)
    nodes = np.vstack([x.ravel(), y.ravel()]).T
    #print ("node.c..", nodes)

    return nodes, radius


def nodesLatLong(numDrones, lat_h, lon_h, coverageDistance, Type):
    if Type == 'T':
        bearing = 45
        nodes, coverageDistance = nodesGeneration_T(numDrones, coverageDistance, Type)
    elif Type == 'L':
        bearing = 90
        nodes, coverageDistance = nodesGeneration_L(numDrones, coverageDistance, 'H')
    elif Type == 'S':
        nodes, coverageDistance = nodesGeneration_S(numDrones, coverageDistance, Type)
        bearing = 45

    elif Type == 'C':
        print (".............c...")
        nodes, coverageDistance = nodesGeneration_C(numDrones, coverageDistance, Type)
        bearing = 45


    # Converting nodes (numpy array) to pandas array for the sake of calculation
    df = pd.DataFrame(data = nodes, columns = ['X','Y'])

    # The initial point of rectangle in (x,y) is (0,0) so considering the current
    # location as origin and retreiving the latitude and longitude from the GPS
    origin = (lat_h, lon_h)

    # Calculating the hypot end point for interpolating the latitudes and longitudes 
    rEndDistance = math.sqrt(2*(coverageDistance**2))

    # The bearing for the hypot angle is 45 degrees considering coverage area as square


    # Determining the Latitude and Longitude of Middle point of the sqaure area
    # and hypot end point of square area for interpolating latitude and longitude
    rMiddle, rEnd = destination_location(origin[0], origin[1], rEndDistance/2, bearing), destination_location(origin[0], origin[1], rEndDistance, bearing)

    # Array of (x,y)
    x_cart, y_cart  = [0, coverageDistance/2, coverageDistance], [0, coverageDistance/2, coverageDistance]

    # Array of (latitude, longitude)
    x_lon, y_lat = [origin[1], rMiddle[1], rEnd[1]], [origin[0], rMiddle[0], rEnd[0]]

    # Latitude interpolation function 
    f_lat = interpolate.interp1d(y_cart, y_lat)

    # Longitude interpolation function
    f_lon = interpolate.interp1d(x_cart, x_lon)

    # Splitting the columns of dataframe (nodes) as x and y
    x, y = df.loc[:,'X'], df.loc[:,'Y']

    #print ("x, y", x, y)

    return (x,y)

def new_gps_coord_after_offset_inBodyFrame(lat_e, lon_e, displacement, current_heading, rotation_degree_relative):
    # current_heading is in degree, North = 0, East = 90.
    # Get rotation degree in local frame.
    rotation_degree_absolute = rotation_degree_relative + current_heading
    if rotation_degree_absolute >= 360:
        rotation_degree_absolute -= 360
    geodesicDistance = geopy.distance.GeodesicDistance(meters = displacement)
    original_point = geopy.Point(lat_e, lon_e)
    new_gps_coord = geodesicDistance.destination(point=original_point, bearing=rotation_degree_absolute)
    new_gps_lat = new_gps_coord.latitude
    new_gps_lon = new_gps_coord.longitude
    # If convert float to decimal, round will be accurate, but will take 50% more time. Not necessary.
    #new_gps_lat = decimal.Decimal(new_gps_lat)
    #new_gps_lon = decimal.Decimal(new_gps_lon)
    return (round(new_gps_lat, 7), round(new_gps_lon, 7))

def formation_heading(no_uavs_count, shape_01, lat, lon, heading_degree):
    global self_heal, master
    global all_pos_data, xy_pos, latlon_pos
    global follower_host_tuple
    global no_uavs
    global cradius

    xy_pos = []
    latlon_pos = []

    lat_00 = lat
    lon_00 = lon
    alt_00 = 10
    original_location = [lat_00, lon_00]

    #coverageDistance = 10
    xoffset = xoffset_entry.get()
    coverageDistance = int(20)

    cradius = cradius_entry.get()

    coverageDistance_c = int(cradius)
    
    if shape_01 == 'T':
        print ("Triangle-shape")
        x,y = nodesLatLong(int(no_uavs_count), lat_00, lon_00, coverageDistance, 'T')

    elif shape_01 == 'L':
        print ("Line-shape")
	#print ("int(no_uavs), lat_00, lon_00, coverageDistance", int(no_uavs), lat_00, lon_00, coverageDistance)
        x,y = nodesLatLong(int(no_uavs_count), lat_00, lon_00, coverageDistance, 'L')
        print (x,y)

    elif shape_01 == 'S':  #square
        print ("square-shape")
        x,y = nodesLatLong(int(no_uavs_count), lat_00, lon_00, coverageDistance, 'S')

    elif shape_01 == 'C':  #square
        print ("circle-shape")
        x,y = nodesLatLong(int(no_uavs_count), lat_00, lon_00, coverageDistance_c, 'C')

    if master == 1:
    	heading = int(heading_degree)
    if master == 2:
    	heading = int(heading_degree)
    if master == 3:
    	heading = int(heading_degree)
    if master == 4:
    	heading = int(heading_degree)
    if master == 5:
    	heading = int(heading_degree)
    if master == 6:
    	heading = int(heading_degree)


    for i in range(0, int(no_uavs_count)):  
        X, Y = float(x[i]), float(y[i])    

        r, theta = polar(X, Y)
        #print ("r, theta", r, theta)
        tlat, tlon = new_gps_coord_after_offset_inBodyFrame(lat_00, lon_00, r, heading, theta)
        #tlat, tlon = get_location_metres(lat, lon, dEast,dNorth)
        #........tlat, tlon = get_location_metres(lat_00, lon_00, X, Y)
        #print (".....tlat, tlon....", i, tlat, tlon)
        pos_xy = (x,y)
        pos_latlon = (tlat, tlon)
        xy_pos.append(pos_xy)
        latlon_pos.append(pos_latlon)

                      
    #print (".....xy_pos......", xy_pos)
    #print (".....latlon_pos......", latlon_pos)

def formation(no_uavs_count, shape_01, lat, lon):
    global self_heal, master
    global all_pos_data, xy_pos, latlon_pos
    global follower_host_tuple
    global no_uavs
    global cradius

    xy_pos = []
    latlon_pos = []

    lat_00 = lat
    lon_00 = lon
    alt_00 = 10
    original_location = [lat_00, lon_00]

    #coverageDistance = 10
    xoffset = xoffset_entry.get()
    coverageDistance = int(xoffset)

    cradius = cradius_entry.get()

    coverageDistance_c = int(cradius)
    
    if shape_01 == 'T':
        print ("Triangle-shape")
        x,y = nodesLatLong(int(no_uavs_count), lat_00, lon_00, coverageDistance, 'T')

    elif shape_01 == 'L':
        print ("Line-shape")
	#print ("int(no_uavs), lat_00, lon_00, coverageDistance", int(no_uavs), lat_00, lon_00, coverageDistance)
        x,y = nodesLatLong(int(no_uavs_count), lat_00, lon_00, coverageDistance, 'L')
        print (x,y)

    elif shape_01 == 'S':  #square
        print ("square-shape")
        x,y = nodesLatLong(int(no_uavs_count), lat_00, lon_00, coverageDistance, 'S')

    elif shape_01 == 'C':  #square
        print ("circle-shape")
        x,y = nodesLatLong(int(no_uavs_count), lat_00, lon_00, coverageDistance_c, 'C')

    if master == 1:
    	heading = int(vehicle1.heading)
    if master == 2:
    	heading = int(vehicle2.heading)
    if master == 3:
    	heading = int(vehicle3.heading)
    if master == 4:
    	heading = int(vehicle4.heading)
    if master == 5:
    	heading = int(vehicle5.heading)
    if master == 6:
    	heading = int(vehicle6.heading)
    	
    heading = heading - 90	
    if heading > 360:
        heading -= 360.0


    for i in range(0, int(no_uavs_count)):  
        X, Y = float(x[i]), float(y[i])    

        r, theta = polar(X, Y)
        #print ("r, theta", r, theta)
        tlat, tlon = new_gps_coord_after_offset_inBodyFrame(lat_00, lon_00, r, heading, theta)
        #tlat, tlon = get_location_metres(lat, lon, dEast,dNorth)
        #........tlat, tlon = get_location_metres(lat_00, lon_00, X, Y)
        #print (".....tlat, tlon....", i, tlat, tlon)
        pos_xy = (x,y)
        pos_latlon = (tlat, tlon)
        xy_pos.append(pos_xy)
        latlon_pos.append(pos_latlon)

                      
    #print (".....xy_pos......", xy_pos)
    #print (".....latlon_pos......", latlon_pos)

def circle():
    global circle_pos_flag
    circle_pos_flag = True



def Group_1():
	global follower_host_tuple_G1,follower_host_tuple_G2,follower_host_tuple_G3,follower_host_tuple_G4,follower_host_tuple_G5
	global counter_G1,counter_G2,counter_G3,counter_G4,counter_G5
	counter_G1 = 0
	follower_host_tuple_G1 = []
	if checkboxvalue_G1_1.get() == 1:
		follower_host_tuple_G1.append(vehicle1)
		counter_G1 = counter_G1+1
	if checkboxvalue_G1_2.get() == 1:
		follower_host_tuple_G1.append(vehicle2)
		counter_G1 = counter_G1+1
	if checkboxvalue_G1_3.get() == 1:
		follower_host_tuple_G1.append(vehicle3)
		counter_G1 = counter_G1+1
	if checkboxvalue_G1_4.get() == 1:
		follower_host_tuple_G1.append(vehicle4)
		counter_G1 = counter_G1+1
	if checkboxvalue_G1_5.get() == 1:
		follower_host_tuple_G1.append(vehicle5)
		counter_G1 = counter_G1+1
	if checkboxvalue_G1_6.get() == 1:
		follower_host_tuple_G1.append(vehicle6)
		counter_G1 = counter_G1+1
	if checkboxvalue_G1_7.get() == 1:
		follower_host_tuple_G1.append(vehicle7)
		counter_G1 = counter_G1+1
	if checkboxvalue_G1_8.get() == 1:
		follower_host_tuple_G1.append(vehicle8)
		counter_G1 = counter_G1+1
	if checkboxvalue_G1_9.get() == 1:
		follower_host_tuple_G1.append(vehicle9)
		counter_G1 = counter_G1+1
	if checkboxvalue_G1_10.get() == 1:
		follower_host_tuple_G1.append(vehicle10)
		counter_G1 = counter_G1+1
	if checkboxvalue_G1_11.get() == 1:
		follower_host_tuple_G1.append(vehicle11)
		counter_G1 = counter_G1+1
	if checkboxvalue_G1_12.get() == 1:
		follower_host_tuple_G1.append(vehicle12)
		counter_G1 = counter_G1+1
	if checkboxvalue_G1_13.get() == 1:
		follower_host_tuple_G1.append(vehicle13)
		counter_G1 = counter_G1+1
	if checkboxvalue_G1_14.get() == 1:
		follower_host_tuple_G1.append(vehicle14)
		counter_G1 = counter_G1+1
	if checkboxvalue_G1_15.get() == 1:
		follower_host_tuple_G1.append(vehicle15)
		counter_G1 = counter_G1+1
	if checkboxvalue_G1_16.get() == 1:
		follower_host_tuple_G1.append(vehicle16)
		counter_G1 = counter_G1+1
	if checkboxvalue_G1_17.get() == 1:
		follower_host_tuple_G1.append(vehicle17)
		counter_G1 = counter_G1+1
	if checkboxvalue_G1_18.get() == 1:
		follower_host_tuple_G1.append(vehicle18)
		counter_G1 = counter_G1+1
	if checkboxvalue_G1_19.get() == 1:
		follower_host_tuple_G1.append(vehicle19)
		counter_G1 = counter_G1+1
	if checkboxvalue_G1_20.get() == 1:
		follower_host_tuple_G1.append(vehicle20)
		counter_G1 = counter_G1+1
	if checkboxvalue_G1_21.get() == 1:
		follower_host_tuple_G1.append(vehicle21)
		counter_G1 = counter_G1+1
	if checkboxvalue_G1_22.get() == 1:
		follower_host_tuple_G1.append(vehicle22)
		counter_G1 = counter_G1+1
	if checkboxvalue_G1_23.get() == 1:
		follower_host_tuple_G1.append(vehicle23)
		counter_G1 = counter_G1+1
	if checkboxvalue_G1_24.get() == 1:
		follower_host_tuple_G1.append(vehicle24)
		counter_G1 = counter_G1+1
	if checkboxvalue_G1_25.get() == 1:
		follower_host_tuple_G1.append(vehicle25)
		counter_G1 = counter_G1+1

	print ("follower_host_tuple_G1", follower_host_tuple_G1)
	print ("counter_G1", counter_G1)

def Group_2():
	global follower_host_tuple_G1,follower_host_tuple_G2,follower_host_tuple_G3,follower_host_tuple_G4,follower_host_tuple_G5
	global counter_G1,counter_G2,counter_G3,counter_G4,counter_G5
	counter_G2 = 0
	follower_host_tuple_G2 = []
	if checkboxvalue_G2_1.get() == 1:
		follower_host_tuple_G2.append(vehicle1)
		counter_G2 = counter_G2+1
	if checkboxvalue_G2_2.get() == 1:
		follower_host_tuple_G2.append(vehicle2)
		counter_G2 = counter_G2+1
	if checkboxvalue_G2_3.get() == 1:
		follower_host_tuple_G2.append(vehicle3)
		counter_G2 = counter_G2+1
	if checkboxvalue_G2_4.get() == 1:
		follower_host_tuple_G2.append(vehicle4)
		counter_G2 = counter_G2+1
	if checkboxvalue_G2_5.get() == 1:
		follower_host_tuple_G2.append(vehicle5)
		counter_G2 = counter_G2+1
	if checkboxvalue_G2_6.get() == 1:
		follower_host_tuple_G2.append(vehicle6)
		counter_G2 = counter_G2+1
	if checkboxvalue_G2_7.get() == 1:
		follower_host_tuple_G2.append(vehicle7)
		counter_G2 = counter_G2+1
	if checkboxvalue_G2_8.get() == 1:
		follower_host_tuple_G2.append(vehicle8)
		counter_G2 = counter_G2+1
	if checkboxvalue_G2_9.get() == 1:
		follower_host_tuple_G2.append(vehicle9)
		counter_G2 = counter_G2+1
	if checkboxvalue_G2_10.get() == 1:
		follower_host_tuple_G2.append(vehicle10)
		counter_G2 = counter_G2+1
	if checkboxvalue_G2_11.get() == 1:
		follower_host_tuple_G2.append(vehicle11)
		counter_G2 = counter_G2+1
	if checkboxvalue_G2_12.get() == 1:
		follower_host_tuple_G2.append(vehicle12)
		counter_G2 = counter_G2+1
	if checkboxvalue_G2_13.get() == 1:
		follower_host_tuple_G2.append(vehicle13)
		counter_G2 = counter_G2+1
	if checkboxvalue_G2_14.get() == 1:
		follower_host_tuple_G2.append(vehicle14)
		counter_G2 = counter_G2+1
	if checkboxvalue_G2_15.get() == 1:
		follower_host_tuple_G2.append(vehicle15)
		counter_G2 = counter_G2+1
	if checkboxvalue_G2_16.get() == 1:
		follower_host_tuple_G2.append(vehicle16)
		counter_G2 = counter_G2+1
	if checkboxvalue_G2_17.get() == 1:
		follower_host_tuple_G2.append(vehicle17)
		counter_G2 = counter_G2+1
	if checkboxvalue_G2_18.get() == 1:
		follower_host_tuple_G2.append(vehicle18)
		counter_G2 = counter_G2+1
	if checkboxvalue_G2_19.get() == 1:
		follower_host_tuple_G2.append(vehicle19)
		counter_G2 = counter_G2+1
	if checkboxvalue_G2_20.get() == 1:
		follower_host_tuple_G2.append(vehicle20)
		counter_G2 = counter_G2+1
	if checkboxvalue_G2_21.get() == 1:
		follower_host_tuple_G2.append(vehicle21)
		counter_G2 = counter_G2+1
	if checkboxvalue_G2_22.get() == 1:
		follower_host_tuple_G2.append(vehicle22)
		counter_G2 = counter_G2+1
	if checkboxvalue_G2_23.get() == 1:
		follower_host_tuple_G2.append(vehicle23)
		counter_G2 = counter_G2+1
	if checkboxvalue_G2_24.get() == 1:
		follower_host_tuple_G2.append(vehicle24)
		counter_G2 = counter_G2+1
	if checkboxvalue_G2_25.get() == 1:
		follower_host_tuple_G2.append(vehicle25)
		counter_G2 = counter_G2+1
	print ("follower_host_tuple_G2", follower_host_tuple_G2)
	print ("counter_G2", counter_G2)


def Group_3():
	global follower_host_tuple_G1,follower_host_tuple_G2,follower_host_tuple_G3,follower_host_tuple_G4,follower_host_tuple_G5
	global counter_G1,counter_G2,counter_G3,counter_G4,counter_G5
	counter_G3 = 0
	follower_host_tuple_G3 = []
	if checkboxvalue_G3_1.get() == 1:
		follower_host_tuple_G3.append(vehicle1)
		counter_G3 = counter_G3+1
	if checkboxvalue_G3_2.get() == 1:
		follower_host_tuple_G3.append(vehicle2)
		counter_G3 = counter_G3+1
	if checkboxvalue_G3_3.get() == 1:
		follower_host_tuple_G3.append(vehicle3)
		counter_G3 = counter_G3+1
	if checkboxvalue_G3_4.get() == 1:
		follower_host_tuple_G3.append(vehicle4)
		counter_G3 = counter_G3+1
	if checkboxvalue_G3_5.get() == 1:
		follower_host_tuple_G3.append(vehicle5)
		counter_G3 = counter_G3+1
	if checkboxvalue_G3_6.get() == 1:
		follower_host_tuple_G3.append(vehicle6)
		counter_G3 = counter_G3+1
	if checkboxvalue_G3_7.get() == 1:
		follower_host_tuple_G3.append(vehicle7)
		counter_G3 = counter_G3+1
	if checkboxvalue_G3_8.get() == 1:
		follower_host_tuple_G3.append(vehicle8)
		counter_G3 = counter_G3+1
	if checkboxvalue_G3_9.get() == 1:
		follower_host_tuple_G3.append(vehicle9)
		counter_G3 = counter_G3+1
	if checkboxvalue_G3_10.get() == 1:
		follower_host_tuple_G3.append(vehicle10)
		counter_G3 = counter_G3+1
	if checkboxvalue_G3_11.get() == 1:
		follower_host_tuple_G3.append(vehicle11)
		counter_G3 = counter_G3+1
	if checkboxvalue_G3_12.get() == 1:
		follower_host_tuple_G3.append(vehicle12)
		counter_G3 = counter_G3+1
	if checkboxvalue_G3_13.get() == 1:
		follower_host_tuple_G3.append(vehicle13)
		counter_G3 = counter_G3+1
	if checkboxvalue_G3_14.get() == 1:
		follower_host_tuple_G3.append(vehicle14)
		counter_G3 = counter_G3+1
	if checkboxvalue_G3_15.get() == 1:
		follower_host_tuple_G3.append(vehicle15)
		counter_G3 = counter_G3+1
	if checkboxvalue_G3_16.get() == 1:
		follower_host_tuple_G3.append(vehicle16)
		counter_G3 = counter_G3+1
	if checkboxvalue_G3_17.get() == 1:
		follower_host_tuple_G3.append(vehicle17)
		counter_G3 = counter_G3+1
	if checkboxvalue_G3_18.get() == 1:
		follower_host_tuple_G3.append(vehicle18)
		counter_G3 = counter_G3+1
	if checkboxvalue_G3_19.get() == 1:
		follower_host_tuple_G3.append(vehicle19)
		counter_G3 = counter_G3+1
	if checkboxvalue_G3_20.get() == 1:
		follower_host_tuple_G3.append(vehicle20)
		counter_G3 = counter_G3+1
	if checkboxvalue_G3_21.get() == 1:
		follower_host_tuple_G3.append(vehicle21)
		counter_G3 = counter_G3+1
	if checkboxvalue_G3_22.get() == 1:
		follower_host_tuple_G3.append(vehicle22)
		counter_G3 = counter_G3+1
	if checkboxvalue_G3_23.get() == 1:
		follower_host_tuple_G3.append(vehicle23)
		counter_G3 = counter_G3+1
	if checkboxvalue_G3_24.get() == 1:
		follower_host_tuple_G3.append(vehicle24)
		counter_G3 = counter_G3+1
	if checkboxvalue_G3_25.get() == 1:
		follower_host_tuple_G3.append(vehicle25)
		counter_G3 = counter_G3+1
	print ("follower_host_tuple_G3", follower_host_tuple_G3)
	print ("counter_G3", counter_G3)


def Group_4():
	global follower_host_tuple_G1,follower_host_tuple_G2,follower_host_tuple_G3,follower_host_tuple_G4,follower_host_tuple_G5
	global counter_G1,counter_G2,counter_G3,counter_G4,counter_G5
	counter_G4 = 0
	follower_host_tuple_G4 = []
	if checkboxvalue_G4_1.get() == 1:
		follower_host_tuple_G4.append(vehicle1)
		counter_G4 = counter_G4+1
	if checkboxvalue_G4_2.get() == 1:
		follower_host_tuple_G4.append(vehicle2)
		counter_G4 = counter_G4+1
	if checkboxvalue_G4_3.get() == 1:
		follower_host_tuple_G4.append(vehicle3)
		counter_G4 = counter_G4+1
	if checkboxvalue_G4_4.get() == 1:
		follower_host_tuple_G4.append(vehicle4)
		counter_G4 = counter_G4+1
	if checkboxvalue_G4_5.get() == 1:
		follower_host_tuple_G4.append(vehicle5)
		counter_G4 = counter_G4+1
	if checkboxvalue_G4_6.get() == 1:
		follower_host_tuple_G4.append(vehicle6)
		counter_G4 = counter_G4+1
	if checkboxvalue_G4_7.get() == 1:
		follower_host_tuple_G4.append(vehicle7)
		counter_G4 = counter_G4+1
	if checkboxvalue_G4_8.get() == 1:
		follower_host_tuple_G4.append(vehicle8)
		counter_G4 = counter_G4+1
	if checkboxvalue_G4_9.get() == 1:
		follower_host_tuple_G4.append(vehicle9)
		counter_G4 = counter_G4+1
	if checkboxvalue_G4_10.get() == 1:
		follower_host_tuple_G4.append(vehicle10)
		counter_G4 = counter_G4+1
	if checkboxvalue_G4_11.get() == 1:
		follower_host_tuple_G4.append(vehicle11)
		counter_G4 = counter_G4+1
	if checkboxvalue_G4_12.get() == 1:
		follower_host_tuple_G4.append(vehicle12)
		counter_G4 = counter_G4+1
	if checkboxvalue_G4_13.get() == 1:
		follower_host_tuple_G4.append(vehicle13)
		counter_G4 = counter_G4+1
	if checkboxvalue_G4_14.get() == 1:
		follower_host_tuple_G4.append(vehicle14)
		counter_G4 = counter_G4+1
	if checkboxvalue_G4_15.get() == 1:
		follower_host_tuple_G4.append(vehicle15)
		counter_G4 = counter_G4+1
	if checkboxvalue_G4_16.get() == 1:
		follower_host_tuple_G4.append(vehicle16)
		counter_G4 = counter_G4+1
	if checkboxvalue_G4_17.get() == 1:
		follower_host_tuple_G4.append(vehicle17)
		counter_G4 = counter_G4+1
	if checkboxvalue_G4_18.get() == 1:
		follower_host_tuple_G4.append(vehicle18)
		counter_G4 = counter_G4+1
	if checkboxvalue_G4_19.get() == 1:
		follower_host_tuple_G4.append(vehicle19)
		counter_G4 = counter_G4+1
	if checkboxvalue_G4_20.get() == 1:
		follower_host_tuple_G4.append(vehicle20)
		counter_G4 = counter_G4+1
	if checkboxvalue_G4_21.get() == 1:
		follower_host_tuple_G4.append(vehicle21)
		counter_G4 = counter_G4+1
	if checkboxvalue_G4_22.get() == 1:
		follower_host_tuple_G4.append(vehicle22)
		counter_G4 = counter_G4+1
	if checkboxvalue_G4_23.get() == 1:
		follower_host_tuple_G4.append(vehicle23)
		counter_G4 = counter_G4+1
	if checkboxvalue_G4_24.get() == 1:
		follower_host_tuple_G4.append(vehicle24)
		counter_G4 = counter_G4+1
	if checkboxvalue_G4_25.get() == 1:
		follower_host_tuple_G4.append(vehicle25)
		counter_G4 = counter_G4+1
	print ("follower_host_tuple_G4", follower_host_tuple_G4)
	print ("counter_G4", counter_G4)


def Group_5():
	global follower_host_tuple_G1,follower_host_tuple_G2,follower_host_tuple_G3,follower_host_tuple_G4,follower_host_tuple_G5
	global counter_G1,counter_G2,counter_G3,counter_G4,counter_G5
	counter_G5 = 0
	follower_host_tuple_G5 = []
	if checkboxvalue_G5_1.get() == 1:
		follower_host_tuple_G5.append(vehicle1)
		counter_G5 = counter_G5+1
	if checkboxvalue_G5_2.get() == 1:
		follower_host_tuple_G5.append(vehicle2)
		counter_G5 = counter_G5+1
	if checkboxvalue_G5_3.get() == 1:
		follower_host_tuple_G5.append(vehicle3)
		counter_G5 = counter_G5+1
	if checkboxvalue_G5_4.get() == 1:
		follower_host_tuple_G5.append(vehicle4)
		counter_G5 = counter_G5+1
	if checkboxvalue_G5_5.get() == 1:
		follower_host_tuple_G5.append(vehicle5)
		counter_G5 = counter_G5+1
	if checkboxvalue_G5_6.get() == 1:
		follower_host_tuple_G5.append(vehicle6)
		counter_G5 = counter_G5+1
	if checkboxvalue_G5_7.get() == 1:
		follower_host_tuple_G5.append(vehicle7)
		counter_G5 = counter_G5+1
	if checkboxvalue_G5_8.get() == 1:
		follower_host_tuple_G5.append(vehicle8)
		counter_G5 = counter_G5+1
	if checkboxvalue_G5_9.get() == 1:
		follower_host_tuple_G5.append(vehicle9)
		counter_G5 = counter_G5+1
	if checkboxvalue_G5_10.get() == 1:
		follower_host_tuple_G5.append(vehicle10)
		counter_G5 = counter_G5+1
	if checkboxvalue_G5_11.get() == 1:
		follower_host_tuple_G5.append(vehicle11)
		counter_G5 = counter_G5+1
	if checkboxvalue_G5_12.get() == 1:
		follower_host_tuple_G5.append(vehicle12)
		counter_G5 = counter_G5+1
	if checkboxvalue_G5_13.get() == 1:
		follower_host_tuple_G5.append(vehicle13)
		counter_G5 = counter_G5+1
	if checkboxvalue_G5_14.get() == 1:
		follower_host_tuple_G5.append(vehicle14)
		counter_G5 = counter_G5+1
	if checkboxvalue_G5_15.get() == 1:
		follower_host_tuple_G5.append(vehicle15)
		counter_G5 = counter_G5+1
	if checkboxvalue_G5_16.get() == 1:
		follower_host_tuple_G5.append(vehicle16)
		counter_G5 = counter_G5+1
	if checkboxvalue_G5_17.get() == 1:
		follower_host_tuple_G5.append(vehicle17)
		counter_G5 = counter_G5+1
	if checkboxvalue_G5_18.get() == 1:
		follower_host_tuple_G5.append(vehicle18)
		counter_G5 = counter_G5+1
	if checkboxvalue_G5_19.get() == 1:
		follower_host_tuple_G5.append(vehicle19)
		counter_G5 = counter_G5+1
	if checkboxvalue_G5_20.get() == 1:
		follower_host_tuple_G5.append(vehicle20)
		counter_G5 = counter_G5+1
	if checkboxvalue_G5_21.get() == 1:
		follower_host_tuple_G5.append(vehicle21)
		counter_G5 = counter_G5+1
	if checkboxvalue_G5_22.get() == 1:
		follower_host_tuple_G5.append(vehicle22)
		counter_G5 = counter_G5+1
	if checkboxvalue_G5_23.get() == 1:
		follower_host_tuple_G5.append(vehicle23)
		counter_G5 = counter_G5+1
	if checkboxvalue_G5_24.get() == 1:
		follower_host_tuple_G5.append(vehicle24)
		counter_G5 = counter_G5+1
	if checkboxvalue_G5_25.get() == 1:
		follower_host_tuple_G5.append(vehicle25)
		counter_G5 = counter_G5+1

	print ("follower_host_tuple_G5", follower_host_tuple_G5)
	print ("counter_G5", counter_G5)

def send():
    global wp_navigation_guided_forward_flag
    wp_navigation_guided_forward_flag = True

def send_01():
    #global wp_navigation_guided_forward_flag
    global wp_navigation_flag, wp_pos
    #generate_search_misison_1()
    print (".....send.....")
    #wp_navigation_guided_forward_flag = True

    global wp_navigation_flag, aggr_and_rtl_flag, wp_navigation_stop_forward_flag
    global master
    global self_heal_odd, self_heal_even
    global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10
    global follower_host_tuple_pos, follower_host_tuple_neg

    export_mission_filename = 'exportedmission_01.txt'
    aFileName = export_mission_filename
    print("\nReading mission from file: %s" % aFileName)
    print ("waypoint_aggregation")
    wp_pos=[]
    #wp_navigation_stop_forward_flag = False
    with open(aFileName) as f:
	for i, line in enumerate(f):  
	    if i==0:
		if not line.startswith('QGC WPL 110'):
		    raise Exception('File is not supported WP version')
	    elif i==1:
		    print ("first way point reject")
	    else:
	    
		    linearray=line.split('\t')
		    ln_index=int(linearray[0])
		    ln_currentwp=int(linearray[1])
		    ln_frame=int(linearray[2])
		    ln_command=int(linearray[3])
		    ln_param1=float(linearray[4])
		    ln_param2=float(linearray[5])
		    ln_param3=float(linearray[6])
		    ln_param4=float(linearray[7])
		    ln_param5=float(linearray[8])
		    ln_param6=float(linearray[9])
		    ln_param7=float(linearray[10])
		    ln_autocontinue=int(linearray[11].strip())
		    pos_data = (ln_param5, ln_param6, ln_param7)
		    wp_pos.append(pos_data)
    print ("...wp_pos.....",wp_pos)

    #wp_navigation_flag = True


def rece():
    global wp_navigation_guided_return_flag
    wp_navigation_guided_return_flag = True

def rece_01():
    global wp_navigation_guided_return_flag
    #global wp_navigation_return_flag, wp_pos
    print (".......rece..........")
    #wp_navigation_guided_return_flag = True
    global wp_navigation_return_flag, aggr_and_rtl_flag, wp_navigation_stop_return_flag
    global master
    global self_heal_odd, self_heal_even
    global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10
    global follower_host_tuple_pos, follower_host_tuple_neg
    export_mission_filename = 'exportedmission_01.txt'
    aFileName = export_mission_filename
    print("\nReading mission from file: %s" % aFileName)
    print ("waypoint_aggregation")
    wp_pos=[]
    #wp_navigation_stop_return_flag = False
    with open(aFileName) as f:
	for i, line in enumerate(f):  
	    if i==0:
		if not line.startswith('QGC WPL 110'):
		    raise Exception('File is not supported WP version')
	    elif i==1:
		    print ("first way point reject")
	    else:
	    
		    linearray=line.split('\t')
		    ln_index=int(linearray[0])
		    ln_currentwp=int(linearray[1])
		    ln_frame=int(linearray[2])
		    ln_command=int(linearray[3])
		    ln_param1=float(linearray[4])
		    ln_param2=float(linearray[5])
		    ln_param3=float(linearray[6])
		    ln_param4=float(linearray[7])
		    ln_param5=float(linearray[8])
		    ln_param6=float(linearray[9])
		    ln_param7=float(linearray[10])
		    ln_autocontinue=int(linearray[11].strip())
		    pos_data = (ln_param5, ln_param6, ln_param7)
		    wp_pos.append(pos_data)

    #wp_navigation_return_flag = True

def stop_forward():
	global wp_navigation_stop_forward_flag
	#generate_search_misison_1()
	print (".....send.....")
	wp_navigation_stop_forward_flag = True

def stop_return():
	global wp_navigation_stop_return_flag
	print (".......rece..........")
	wp_navigation_stop_return_flag = True


def aggr(): 
        global check_box_flag3       
        global xy_pos, latlon_pos
        global master, no_uavs
        global self_heal
        global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10,vehicle11,vehicle12,vehicle13
        global vehicle14,vehicle15,vehicle16,vehicle17,vehicle18,vehicle19,vehicle20,vehicle21,vehicle22,vehicle23,vehicle24,vehicle25      
	global follower_host_tuple_G1,follower_host_tuple_G2,follower_host_tuple_G3,follower_host_tuple_G4,follower_host_tuple_G5
	global counter_G1,counter_G2,counter_G3,counter_G4,counter_G5

	global follower_host_tuple
        global circle_pos_flag

        xoffset = xoffset_entry.get()
        cradius = cradius_entry.get()
        aoffset = aoffset_entry.get()
        salt = salt_entry.get() 
        xoffset = int(xoffset)
        cradius = int(cradius)
        aoffset = int(aoffset)
        salt = int(salt)
        print ("aggregation", master)
	#................group 1......................
        if checkboxvalue_Group_1.get() == 1:
                Group_1()
                try:
			G1_M = G1_master_set_entry.get()
			G1_M = int(G1_M)
			if G1_M == 1:
			    lat = vehicle1.location.global_relative_frame.lat
			    lon = vehicle1.location.global_relative_frame.lon
			    alt_0 = vehicle1.location.global_relative_frame.alt
			    clat=g_lat1.get()
			    clon=g_lon1.get()
			    clat = float(clat)
			    clon = float(clon)

			if G1_M == 2:
			    lat = vehicle2.location.global_relative_frame.lat
			    lon = vehicle2.location.global_relative_frame.lon
			    alt_0 = vehicle2.location.global_relative_frame.alt
			    clat=g_lat2.get()
			    clon=g_lon2.get()
			    clat = float(clat)
			    clon = float(clon)

			if G1_M == 3:
			    lat = vehicle3.location.global_relative_frame.lat
			    lon = vehicle3.location.global_relative_frame.lon
			    alt_0 = vehicle3.location.global_relative_frame.alt
			    clat=g_lat3.get()
			    clon=g_lon3.get()
			    clat = float(clat)
			    clon = float(clon)


			if checkboxvalue1.get() == 1:
			    formation(counter_G1, 'T', lat, lon)
			elif checkboxvalue2.get() == 1:
			    formation(counter_G1, 'L', lat, lon)
			elif checkboxvalue3.get() == 1:
			    formation(counter_G1, 'S', lat, lon)
			elif checkboxvalue4.get() == 1:
			    """
			    clat = clat_entry.get()
			    clon = clon_entry.get()
			    """

			    formation(counter_G1, 'C', clat, clon)


			print ("................", latlon_pos)
			#for i in range(0, int(no_uavs)):  
			alt_001 = salt
			for i, iter_follower_G1 in enumerate(follower_host_tuple_G1):
			    if iter_follower_G1 == None:
				##print "lost odd uav", self_heal[i]
				print ("payload drop lost  uav", (i+1)) 
				if check_box_flag3 == True: 
					print ("self heal..to alt change")   
				else:
					alt_001 = alt_001
					#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
			    else: 
				#if i < int(no_uavs):   
				print ("payload present  uav :", (i+1))    
				###init_pos1 = altered_position(original_location,a,b)  
				test = latlon_pos[i]
				print ("..t_goto..", test[0], test[1])    
				alt_123 = iter_follower_G1.location.global_relative_frame.alt  
				print ("...............Dhikshith....Group 1......", alt_123)
				target = LocationGlobalRelative(test[0], test[1], alt_123)
				print ("target", target)
				for i in range(0, 5):
					aggregation_formation(iter_follower_G1,"GUIDED", target) 
                except:
			pass
	#................group 2......................  
        if checkboxvalue_Group_2.get() == 1:
                Group_2()
                try:
			G2_M = G2_master_set_entry.get()
			G2_M = int(G2_M)
		 
			if G2_M == 4:
			    lat = vehicle4.location.global_relative_frame.lat
			    lon = vehicle4.location.global_relative_frame.lon
			    alt_0 = vehicle4.location.global_relative_frame.alt
			    clat=g_lat4.get()
			    clon=g_lon4.get()
			    clat = float(clat)
			    clon = float(clon)

			if G2_M == 5:
			    lat = vehicle5.location.global_relative_frame.lat
			    lon = vehicle5.location.global_relative_frame.lon
			    alt_0 = vehicle5.location.global_relative_frame.alt
			    clat=g_lat5.get()
			    clon=g_lon5.get()
			    clat = float(clat)
			    clon = float(clon)

			if G2_M == 6:
			    lat = vehicle6.location.global_relative_frame.lat
			    lon = vehicle6.location.global_relative_frame.lon
			    alt_0 = vehicle6.location.global_relative_frame.alt
			    clat=g_lat6.get()
			    clon=g_lon6.get()
			    clat = float(clat)
			    clon = float(clon)

			if checkboxvalue1.get() == 1:
			    formation(counter_G2, 'T', lat, lon)
			elif checkboxvalue2.get() == 1:
			    formation(counter_G2, 'L', lat, lon)
			elif checkboxvalue3.get() == 1:
			    formation(counter_G2, 'S', lat, lon)
			elif checkboxvalue4.get() == 1:
			    """
			    clat = clat_entry.get()
			    clon = clon_entry.get()
			    """

			    formation(counter_G1, 'C', clat, clon)

			print ("follower_host_tuple_G2", follower_host_tuple_G2)
			print ("counter_G2", counter_G2)
			print ("................", latlon_pos)
			#for i in range(0, int(no_uavs)):  
			alt_001 = salt
			for i, iter_follower_G2 in enumerate(follower_host_tuple_G2):
			    if iter_follower_G2 == None:
				##print "lost odd uav", self_heal[i]
				print ("payload drop lost  uav", (i+1)) 
				if check_box_flag3 == True: 
					print ("self heal..to alt change")   
				else:
					alt_001 = alt_001
					#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
			    else: 
				#if i < int(no_uavs):   
				print ("payload present  uav :", (i+1))    
				###init_pos1 = altered_position(original_location,a,b)  
				test = latlon_pos[i]
				print ("..t_goto..", test[0], test[1])    
				alt_123 = iter_follower_G2.location.global_relative_frame.alt  
				print ("...............Dhikshith..group 2........", alt_123)
				target = LocationGlobalRelative(test[0], test[1], alt_123)
				print ("target", target)
				for i in range(0, 5):
					aggregation_formation(iter_follower_G2,"GUIDED", target)  
                except:
			pass  

	#................group 3......................
        if checkboxvalue_Group_3.get() == 1:
                Group_3()
                try:
			G3_M = G3_master_set_entry.get()
			G3_M = int(G3_M)
			if G3_M == 11:
			    lat = vehicle11.location.global_relative_frame.lat
			    lon = vehicle11.location.global_relative_frame.lon
			    alt_0 = vehicle11.location.global_relative_frame.alt
			    clat=g_lat11.get()
			    clon=g_lon11.get()
			    clat = float(clat)
			    clon = float(clon)

			if G3_M == 12:
			    lat = vehicle12.location.global_relative_frame.lat
			    lon = vehicle12.location.global_relative_frame.lon
			    alt_0 = vehicle12.location.global_relative_frame.alt
			    clat=g_lat12.get()
			    clon=g_lon12.get()
			    clat = float(clat)
			    clon = float(clon)

			if G3_M == 13:
			    lat = vehicle13.location.global_relative_frame.lat
			    lon = vehicle13.location.global_relative_frame.lon
			    alt_0 = vehicle13.location.global_relative_frame.alt
			    clat=g_lat13.get()
			    clon=g_lon13.get()
			    clat = float(clat)
			    clon = float(clon)
				 
			if G3_M == 14:
			    lat = vehicle14.location.global_relative_frame.lat
			    lon = vehicle14.location.global_relative_frame.lon
			    alt_0 = vehicle14.location.global_relative_frame.alt

			if G3_M == 15:
			    lat = vehicle15.location.global_relative_frame.lat
			    lon = vehicle15.location.global_relative_frame.lon
			    alt_0 = vehicle15.location.global_relative_frame.alt
			
			if checkboxvalue1.get() == 1:
			    formation(counter_G3, 'T', lat, lon)
			elif checkboxvalue2.get() == 1:
			    formation(counter_G3, 'L', lat, lon)
			elif checkboxvalue3.get() == 1:
			    formation(counter_G3, 'S', lat, lon)
			elif checkboxvalue4.get() == 1:
			    """
			    clat = clat_entry.get()
			    clon = clon_entry.get()
			    """

			    formation(counter_G1, 'C', clat, clon)

			print ("follower_host_tuple_G3", follower_host_tuple_G3)
			print ("counter_G3", counter_G3)
			print ("................", latlon_pos)
			#for i in range(0, int(no_uavs)):  
			alt_001 = salt
			for i, iter_follower_G3 in enumerate(follower_host_tuple_G3):
			    if iter_follower_G3 == None:
				##print "lost odd uav", self_heal[i]
				print ("payload drop lost  uav", (i+1)) 
				if check_box_flag3 == True: 
					print ("self heal..to alt change")   
				else:
					alt_001 = alt_001
					#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
			    else: 
				#if i < int(no_uavs):   
				print ("payload present  uav :", (i+1))    
				###init_pos1 = altered_position(original_location,a,b)  
				test = latlon_pos[i]
				print ("..t_goto..", test[0], test[1])    
				alt_123 = iter_follower_G3.location.global_relative_frame.alt  
				print ("...............Dhikshith..........", alt_123)
				target = LocationGlobalRelative(test[0], test[1], alt_123)
				print ("target", target)
				for i in range(0, 5):
					aggregation_formation(iter_follower_G3,"GUIDED", target)   
                except:
			pass 
	#................group 4......................
        if checkboxvalue_Group_4.get() == 1:
                Group_4()
                try:
			G1_M = G1_master_set_entry.get()
			G1_M = int(G1_M)

			if checkboxvalue1.get() == 1:
			    formation(counter_G4, 'T', lat, lon)
			elif checkboxvalue2.get() == 1:
			    formation(counter_G4, 'L', lat, lon)
			elif checkboxvalue3.get() == 1:
			    formation(counter_G4, 'S', lat, lon)
			elif checkboxvalue4.get() == 1:
			    if circle_pos_flag == True:
				circle_pos_flag = False
				clat = clat_entry.get()
				clon = clon_entry.get()
				clat = float(clat)
				clon = float(clon)
				formation(counter_G4, 'C', clat, clon)
			    else:
				formation(counter_G4, 'C', lat, lon)

			print ("follower_host_tuple_G4", follower_host_tuple_G4)
			print ("counter_G4", counter_G4)

			print ("................", latlon_pos)
			#for i in range(0, int(no_uavs)):  
			alt_001 = salt
			for i, iter_follower_G4 in enumerate(follower_host_tuple_G4):
			    if iter_follower_G4 == None:
				##print "lost odd uav", self_heal[i]
				print ("payload drop lost  uav", (i+1)) 
				if check_box_flag3 == True: 
					print ("self heal..to alt change")   
				else:
					alt_001 = alt_001
					#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
			    else: 
				#if i < int(no_uavs):   
				print ("payload present  uav :", (i+1))    
				###init_pos1 = altered_position(original_location,a,b)  
				test = latlon_pos[i]
				print ("..t_goto..", test[0], test[1])    
				alt_123 = iter_follower_G4.location.global_relative_frame.alt  
				print ("...............Dhikshith..........", alt_123)
				target = LocationGlobalRelative(test[0], test[1], alt_123)
				print ("target", target)
				for i in range(0, 5):
					aggregation_formation(iter_follower_G4,"GUIDED", target)   
                except:
			pass 

        if checkboxvalue_Group_all.get() == 1:
		print ("all UAV")
		print (".....master.....", master)
        #..............for all uav.................
		if master == 1:
		    lat = vehicle1.location.global_relative_frame.lat
		    lon = vehicle1.location.global_relative_frame.lon
		    alt_0 = vehicle1.location.global_relative_frame.alt

		if master == 2:
		    lat = vehicle2.location.global_relative_frame.lat
		    lon = vehicle2.location.global_relative_frame.lon
		    alt_0 = vehicle2.location.global_relative_frame.alt

		if master == 3:
		    lat = vehicle3.location.global_relative_frame.lat
		    lon = vehicle3.location.global_relative_frame.lon
		    alt_0 = vehicle3.location.global_relative_frame.alt
		         
		if master == 4:
		    lat = vehicle4.location.global_relative_frame.lat
		    lon = vehicle4.location.global_relative_frame.lon
		    alt_0 = vehicle4.location.global_relative_frame.alt

		if master == 5:
		    lat = vehicle5.location.global_relative_frame.lat
		    lon = vehicle5.location.global_relative_frame.lon
		    alt_0 = vehicle5.location.global_relative_frame.alt

		if master == 6:
		    lat = vehicle6.location.global_relative_frame.lat
		    lon = vehicle6.location.global_relative_frame.lon
		    alt_0 = vehicle6.location.global_relative_frame.alt

		if master == 7:
		    lat = vehicle7.location.global_relative_frame.lat
		    lon = vehicle7.location.global_relative_frame.lon
		    alt_0 = vehicle7.location.global_relative_frame.alt

		if master == 8:
		    lat = vehicle8.location.global_relative_frame.lat
		    lon = vehicle8.location.global_relative_frame.lon
		    alt_0 = vehicle8.location.global_relative_frame.alt

		if master == 9:
		    lat = vehicle9.location.global_relative_frame.lat
		    lon = vehicle9.location.global_relative_frame.lon
		    alt_0 = vehicle9.location.global_relative_frame.alt

		if master == 10:
		    lat = vehicle10.location.global_relative_frame.lat
		    lon = vehicle10.location.global_relative_frame.lon
		    alt_0 = vehicle10.location.global_relative_frame.alt

		if checkboxvalue1.get() == 1:
		    formation(int(no_uavs), 'T', lat, lon)
		elif checkboxvalue2.get() == 1:
		    formation(int(no_uavs), 'L', lat, lon)
		elif checkboxvalue3.get() == 1:
		    formation(int(no_uavs), 'S', lat, lon)
		elif checkboxvalue4.get() == 1:
		    if circle_pos_flag == True:
		        circle_pos_flag = False
		        clat = clat_entry.get()
		        clon = clon_entry.get()
		        clat = float(clat)
		        clon = float(clon)
		        formation(int(no_uavs), 'C', clat, clon)
		    else:
		        formation(int(no_uavs), 'C', lat, lon)

	   
		a,b,c = (0,0,0)
		count_wp = 0
		print (follower_host_tuple)
		print ("................", self_heal)
		#for i in range(0, int(no_uavs)):  
		alt_001 = salt
		print ("..self_heal..", self_heal)
		for i, iter_follower in enumerate(follower_host_tuple): 
		    if self_heal[i] > 0:
		        ##print "lost odd uav", self_heal[i]
		        print ("lost  uav", (i+1))
		        pos_latlon = (0.0, 0.0)
		        latlon_pos.insert(i, (0.0, 0.0))
		        ##c = (c+20)   
		        ##alt_001 = alt_001 + aoffset
                        
		        if check_box_flag3 == True: 
		        	print ("self heal..to alt change")   
			else:
		        	alt_001 = alt_001 + aoffset   #....... alt not change during self heal
		 
		    else: 
		        #if i < int(no_uavs):   
		        print ("present  uav :", (i+1))    
		        ###init_pos1 = altered_position(original_location,a,b)  
		        test = latlon_pos[i]
		        print ("..t_goto..", test[0], test[1])        
		        target = LocationGlobalRelative(test[0], test[1], alt_001)
		        print ("target", target)
			for i in range(0, 5):
				if alt_0 >= 10:
					if iter_follower.mode.name == "RTL":
						print ("vehicle RTL")
					else:
		        			aggregation_formation(iter_follower,"GUIDED", target)        
		        alt_001 = alt_001 + aoffset
        



def arm_all():
        global follower_host_tuple
        print ("...........arm_all.........")
        print ("follower_host_tuple", follower_host_tuple)
        for i, iter_follower in enumerate(follower_host_tuple):  
            #if self_heal[i] > 0:
            #    print ("slave is lost")
            #else:        
            iter_follower.armed = True
            time.sleep(0.2)

def disarm_all():
        global follower_host_tuple
        print ("...........disarm_all.........")
        print ("follower_host_tuple", follower_host_tuple)
        for i, iter_follower in enumerate(follower_host_tuple):  
            #if self_heal[i] > 0:
             #   print ("slave is lost")
            #else:        
             iter_follower.armed = False
             time.sleep(0.2)


def guided_all():
        global follower_host_tuple
        print ("...........mode changed.........")
        print ("follower_host_tuple", follower_host_tuple)
        for iter_follower in follower_host_tuple:  
            if iter_follower == None:
                print ("slave is lost")
            else:        
                iter_follower.mode = VehicleMode('GUIDED')
                time.sleep(0.2)

def takeoff_all_ok(iter_follower,takeoff_alt):
        global control_command
	try:
		iter_follower.mode = VehicleMode('GUIDED')
		iter_follower.armed = True
		time.sleep(3)
		if iter_follower.armed == True:
			iter_follower.simple_takeoff(takeoff_alt)
			time.sleep(0.5)
	except:
		pass

def takeoff_all():
        global control_command
	try:
		takeoff_alt = takeoff_entry.get()
		try:
			takeoff_alt = int(takeoff_alt)
		except:
			takeoff_alt = int(10)
		global follower_host_tuple
		print ("...........mode changed.........")
		print ("follower_host_tuple", follower_host_tuple)
		for i, iter_follower in enumerate(follower_host_tuple):  
		    #if iter_follower == None:
		    if self_heal[i] > 0:
		        print ("slave is lost")
		    else:  
			print ("takoff uav")
			threading.Thread(target=takeoff_all_ok,args=(iter_follower,takeoff_alt,)).start()
	except:
		pass




def guided_main():
        global follower_host_tuple_main
        print ("...........mode changed.........")
        print ("follower_host_tuple_main", follower_host_tuple_main)
        for iter_follower_main in follower_host_tuple_main:  
            if iter_follower_main == None:
                print ("slave is lost")
            else:        
                iter_follower_main.mode = VehicleMode('GUIDED')
                time.sleep(0.2)

def guided_sec():
        global follower_host_tuple_sec
        print ("...........mode changed.........")
        print ("follower_host_tuple_sec", follower_host_tuple_sec)
        for iter_follower_sec in follower_host_tuple_sec:  
            if iter_follower_sec == None:
                print ("slave is lost")
            else:        
                iter_follower_sec.mode = VehicleMode('GUIDED')
                time.sleep(0.2)

def pause():
    global check_box_flag3       
    global xy_pos, latlon_pos
    global master, no_uavs
    global self_heal
    global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10,vehicle11,vehicle12,vehicle13
    global vehicle14,vehicle15,vehicle16,vehicle17,vehicle18,vehicle19,vehicle20,vehicle21,vehicle22,vehicle23,vehicle24,vehicle25      
    global follower_host_tuple_G1,follower_host_tuple_G2,follower_host_tuple_G3,follower_host_tuple_G4,follower_host_tuple_G5
    global counter_G1,counter_G2,counter_G3,counter_G4,counter_G5
    
    global follower_host_tuple
    global circle_pos_flag
    global RTL_all_flag
    xoffset = xoffset_entry.get()
    cradius = cradius_entry.get()
    aoffset = aoffset_entry.get()
    salt = salt_entry.get() 
    xoffset = int(xoffset)
    cradius = int(cradius)
    aoffset = int(aoffset)
    salt = int(salt)
    RTL_all_flag = False
    if checkboxvalue_Group_1.get() == 1:
	Group_1()
	alt_001 = salt
	for i, iter_follower_G1 in enumerate(follower_host_tuple_G1):
	    if iter_follower_G1 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		print ("payload present  uav :", (i+1)) 
		iter_follower_G1.mode = VehicleMode('GUIDED')
		time.sleep(0.2)
    if checkboxvalue_Group_2.get() == 1:
	Group_2()
	alt_001 = salt
	for i, iter_follower_G2 in enumerate(follower_host_tuple_G2):
	    if iter_follower_G2 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		print ("payload present  uav :", (i+1)) 
		iter_follower_G2.mode = VehicleMode('GUIDED')
		time.sleep(0.2)
    if checkboxvalue_Group_3.get() == 1:
	Group_3()
	alt_001 = salt
	for i, iter_follower_G3 in enumerate(follower_host_tuple_G3):
	    if iter_follower_G3 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		print ("payload present  uav :", (i+1)) 
		iter_follower_G3.mode = VehicleMode('GUIDED')
		time.sleep(0.2)
    if checkboxvalue_Group_4.get() == 1:
	Group_4()
	alt_001 = salt
	for i, iter_follower_G4 in enumerate(follower_host_tuple_G4):
	    if iter_follower_G4 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		print ("payload present  uav :", (i+1)) 
		iter_follower_G4.mode = VehicleMode('GUIDED')
		time.sleep(0.2)

    if checkboxvalue_Group_all.get() == 1:
	    #for iter_follower in follower_host_tuple: 
	    for i, iter_follower in enumerate(follower_host_tuple):
		if self_heal[i] > 0:
		    print ("slave is lost")
		else:       
		    iter_follower.mode = VehicleMode('GUIDED')
		    time.sleep(0.2)

def heading():
    global check_box_flag3       
    global xy_pos, latlon_pos
    global master, no_uavs
    global self_heal
    global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10,vehicle11,vehicle12,vehicle13
    global vehicle14,vehicle15,vehicle16,vehicle17,vehicle18,vehicle19,vehicle20,vehicle21,vehicle22,vehicle23,vehicle24,vehicle25      
    global follower_host_tuple_G1,follower_host_tuple_G2,follower_host_tuple_G3,follower_host_tuple_G4,follower_host_tuple_G5
    global counter_G1,counter_G2,counter_G3,counter_G4,counter_G5

    global follower_host_tuple
    global circle_pos_flag
    xoffset = xoffset_entry.get()
    cradius = cradius_entry.get()
    aoffset = aoffset_entry.get()
    salt = salt_entry.get() 
    xoffset = int(xoffset)
    cradius = int(cradius)
    aoffset = int(aoffset)
    salt = int(salt)
    if checkboxvalue_Group_1.get() == 1:
	Group_1()
	alt_001 = salt
	for i, iter_follower_G1 in enumerate(follower_host_tuple_G1):
	    if iter_follower_G1 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		    print ("payload present  uav :", (i+1)) 
		    heading_01 = headingset_entry.get()
		    heading_01 = int(heading_01)
		    relative=False
		    if relative:
			is_relative = 1 #yaw relative to direction of travel
		    else:
			is_relative = 0 #yaw is an absolute angle
		    # create the CONDITION_YAW command using command_long_encode()
		    msg = iter_follower_G1.message_factory.command_long_encode(
			0, 0,    # target system, target component
			mavutil.mavlink.MAV_CMD_CONDITION_YAW, #command
			0, #confirmation
			heading_01,    # param 1, yaw in degrees
			0,          # param 2, yaw speed deg/s
			1,          # param 3, direction -1 ccw, 1 cw
			is_relative, # param 4, relative offset 1, absolute angle 0
			0, 0, 0)    # param 5 ~ 7 not used
		    # send command to vehicle
		    iter_follower_G1.send_mavlink(msg)
		    time.sleep(0.1)
    if checkboxvalue_Group_2.get() == 1:
	Group_2()
	alt_001 = salt
	for i, iter_follower_G2 in enumerate(follower_host_tuple_G2):
	    if iter_follower_G2 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		    print ("payload present  uav :", (i+1)) 
		    heading_01 = headingset_entry.get()
		    heading_01 = int(heading_01)
		    relative=False
		    if relative:
			is_relative = 1 #yaw relative to direction of travel
		    else:
			is_relative = 0 #yaw is an absolute angle
		    # create the CONDITION_YAW command using command_long_encode()
		    msg = iter_follower_G2.message_factory.command_long_encode(
			0, 0,    # target system, target component
			mavutil.mavlink.MAV_CMD_CONDITION_YAW, #command
			0, #confirmation
			heading_01,    # param 1, yaw in degrees
			0,          # param 2, yaw speed deg/s
			1,          # param 3, direction -1 ccw, 1 cw
			is_relative, # param 4, relative offset 1, absolute angle 0
			0, 0, 0)    # param 5 ~ 7 not used
		    # send command to vehicle
		    iter_follower_G2.send_mavlink(msg)
		    time.sleep(0.1)
    if checkboxvalue_Group_3.get() == 1:
	Group_3()
	alt_001 = salt
	for i, iter_follower_G3 in enumerate(follower_host_tuple_G3):
	    if iter_follower_G3 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		    print ("payload present  uav :", (i+1)) 
		    heading_01 = headingset_entry.get()
		    heading_01 = int(heading_01)
		    relative=False
		    if relative:
			is_relative = 1 #yaw relative to direction of travel
		    else:
			is_relative = 0 #yaw is an absolute angle
		    # create the CONDITION_YAW command using command_long_encode()
		    msg = iter_follower_G3.message_factory.command_long_encode(
			0, 0,    # target system, target component
			mavutil.mavlink.MAV_CMD_CONDITION_YAW, #command
			0, #confirmation
			heading_01,    # param 1, yaw in degrees
			0,          # param 2, yaw speed deg/s
			1,          # param 3, direction -1 ccw, 1 cw
			is_relative, # param 4, relative offset 1, absolute angle 0
			0, 0, 0)    # param 5 ~ 7 not used
		    # send command to vehicle
		    iter_follower_G3.send_mavlink(msg)
		    time.sleep(0.1)
    if checkboxvalue_Group_4.get() == 1:
	Group_4()
	alt_001 = salt
	for i, iter_follower_G4 in enumerate(follower_host_tuple_G4):
	    if iter_follower_G4 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		    print ("payload present  uav :", (i+1)) 
		    heading_01 = headingset_entry.get()
		    heading_01 = int(heading_01)
		    relative=False
		    if relative:
			is_relative = 1 #yaw relative to direction of travel
		    else:
			is_relative = 0 #yaw is an absolute angle
		    # create the CONDITION_YAW command using command_long_encode()
		    msg = iter_follower_G4.message_factory.command_long_encode(
			0, 0,    # target system, target component
			mavutil.mavlink.MAV_CMD_CONDITION_YAW, #command
			0, #confirmation
			heading_01,    # param 1, yaw in degrees
			0,          # param 2, yaw speed deg/s
			1,          # param 3, direction -1 ccw, 1 cw
			is_relative, # param 4, relative offset 1, absolute angle 0
			0, 0, 0)    # param 5 ~ 7 not used
		    # send command to vehicle
		    iter_follower_G4.send_mavlink(msg)
		    time.sleep(0.1)

    if checkboxvalue_Group_all.get() == 1:
	    #for iter_follower in follower_host_tuple: 
	    for i, iter_follower in enumerate(follower_host_tuple):
		if self_heal[i] > 0:
		    print ("slave is lost")
		else:       
		    heading_01 = headingset_entry.get()
		    heading_01 = int(heading_01)
		    relative=False
		    if relative:
			is_relative = 1 #yaw relative to direction of travel
		    else:
			is_relative = 0 #yaw is an absolute angle
		    # create the CONDITION_YAW command using command_long_encode()
		    msg = iter_follower.message_factory.command_long_encode(
			0, 0,    # target system, target component
			mavutil.mavlink.MAV_CMD_CONDITION_YAW, #command
			0, #confirmation
			heading_01,    # param 1, yaw in degrees
			0,          # param 2, yaw speed deg/s
			1,          # param 3, direction -1 ccw, 1 cw
			is_relative, # param 4, relative offset 1, absolute angle 0
			0, 0, 0)    # param 5 ~ 7 not used
		    # send command to vehicle
		    iter_follower.send_mavlink(msg)

def heading_123(vehicle,heading_01):
    global follower_host_tuple
    print ("...........heading changed.........")

    relative=False

    if relative:
	is_relative = 1 #yaw relative to direction of travel
    else:
	is_relative = 0 #yaw is an absolute angle
    # create the CONDITION_YAW command using command_long_encode()
    msg = vehicle.message_factory.command_long_encode(
	0, 0,    # target system, target component
	mavutil.mavlink.MAV_CMD_CONDITION_YAW, #command
	0, #confirmation
	heading_01,    # param 1, yaw in degrees
	0,          # param 2, yaw speed deg/s
	1,          # param 3, direction -1 ccw, 1 cw
	is_relative, # param 4, relative offset 1, absolute angle 0
	0, 0, 0)    # param 5 ~ 7 not used
    # send command to vehicle
    vehicle.send_mavlink(msg)


def land():
    global check_box_flag3       
    global xy_pos, latlon_pos
    global master, no_uavs
    global self_heal
    global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10,vehicle11,vehicle12,vehicle13
    global vehicle14,vehicle15,vehicle16,vehicle17,vehicle18,vehicle19,vehicle20,vehicle21,vehicle22,vehicle23,vehicle24,vehicle25      
    global follower_host_tuple_G1,follower_host_tuple_G2,follower_host_tuple_G3,follower_host_tuple_G4,follower_host_tuple_G5
    global counter_G1,counter_G2,counter_G3,counter_G4,counter_G5

    global follower_host_tuple
    print("follower_host_tuple",follower_host_tuple)
    global circle_pos_flag
    xoffset = xoffset_entry.get()
    cradius = cradius_entry.get()
    aoffset = aoffset_entry.get()
    salt = salt_entry.get() 
    xoffset = int(xoffset)
    cradius = int(cradius)
    aoffset = int(aoffset)
    salt = int(salt)
    if checkboxvalue_Group_1.get() == 1:
	Group_1()
	alt_001 = salt
	for i, iter_follower_G1 in enumerate(follower_host_tuple_G1):
	    if iter_follower_G1 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		print ("payload present  uav :", (i+1)) 
		iter_follower_G1.mode = VehicleMode('LAND')
		time.sleep(0.2)
    if checkboxvalue_Group_2.get() == 1:
	Group_2()
	alt_001 = salt
	for i, iter_follower_G2 in enumerate(follower_host_tuple_G2):
	    if iter_follower_G2 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		print ("payload present  uav :", (i+1)) 
		iter_follower_G2.mode = VehicleMode('LAND')
		time.sleep(0.2)
    if checkboxvalue_Group_3.get() == 1:
	Group_3()
	alt_001 = salt
	for i, iter_follower_G3 in enumerate(follower_host_tuple_G3):
	    if iter_follower_G3 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		print ("payload present  uav :", (i+1)) 
		iter_follower_G3.mode = VehicleMode('LAND')
		time.sleep(0.2)
    if checkboxvalue_Group_4.get() == 1:
	Group_4()
	alt_001 = salt
	for i, iter_follower_G4 in enumerate(follower_host_tuple_G4):
	    if iter_follower_G4 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		print ("payload present  uav :", (i+1)) 
		iter_follower_G4.mode = VehicleMode('LAND')
		time.sleep(0.2)

    if checkboxvalue_Group_all.get() == 1:
	    #for iter_follower in follower_host_tuple: 
	    for i, iter_follower in enumerate(follower_host_tuple):
		if self_heal[i] > 0:
		    print ("slave is lost")
		else:       
		    iter_follower.mode = VehicleMode('LAND')
		    time.sleep(0.2)

def payload_all():
        global follower_host_tuple
        for iter_follower in follower_host_tuple:  
            if iter_follower == None:
                print ("slave is lost")
            else:        
                payload_send(iter_follower)
                #time.sleep(0.1)


def payload_send(vehicle):
    if vehicle == None:
            print ("vehicle is lost")
    else:
    #...........payload.........
        ###servo_test(vehicle, 1100, 1)
        ##time.sleep(0.5)
        for i in range(0, 3):
            servo_test(vehicle, 1900, 2)
            time.sleep(0.1)
        #servo_test(vehicle, 1100, 1)
        #time.sleep(0.5)

def set_servo(vehicle, servo_number, pwm_value):

        pwm_value_int = int(pwm_value)
        msg = vehicle.message_factory.command_long_encode(
                0, 0, # target system, target component
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                0,
                servo_number,
                pwm_value_int,
                0,0,0,0,0
                )
        vehicle.send_mavlink(msg)

def servo_test(vehicle, pwm, check):

    #print("servo to %s" % pwm)
    set_servo(vehicle, 10, pwm)
    if check == 2:
        print("payload drop successfully")


def C_Radius():
    global cradius
    cradius = cradius_entry.get()
    cradius = int(cradius)

def circlePOI(vehicle_1, speed, radius):
    # The circle radius in cm. Max 10000
    # The tangential speed is 50 cm/s


    # Radius has to be increments of 100 cm and rate has to be in increments of 1 degree
    radius = int(radius)
    period = 2*math.pi*radius / speed
    rate = int(360.0/period)
    print ("...period, rate", period, rate) 

    vehicle_1.parameters["CIRCLE_RADIUS"] = radius
    vehicle_1.parameters["CIRCLE_RATE"] = rate

    vehicle_1.mode = VehicleMode("CIRCLE")
    #time.sleep(1)
    vehicle_1.channels.overrides['3'] = 1500
    time.sleep(0.5)

def set_roi(vehicle, clat, clon, calt):
    """
    Send MAV_CMD_DO_SET_ROI message to point camera gimbal at a 
    specified region of interest (LocationGlobal).
    The vehicle may also turn to face the ROI.

    For more information see: 
    http://copter.ardupilot.com/common-mavlink-mission-command-messages-mav_cmd/#mav_cmd_do_set_roi
    """

    # create the MAV_CMD_DO_SET_ROI command
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_DO_SET_ROI, #command
        0, #confirmation
        0, 0, 0, 0, #params 1-4
        clat,
        clon,
        calt
        )
    # send command to vehicle
    vehicle.send_mavlink(msg)


def curvature_flight_body_frame(vehicle, horizontal_linear_speed, radius_of_curvature, total_turn_degree_deg, velocity_z, atom_segment):

    print('\n')
    print('{} - Calling function curvature_flight_body_frame(horizontal_linear_speed={}, radius_of_curvature={}, velocity_z={})'.format(time.ctime(), horizontal_linear_speed, radius_of_curvature, velocity_z))
    # Get sign of radius_of_curvature. # Positive: turn right. Negative: turn left.
    turn_direction = np.sign(radius_of_curvature)
    # Convert int to float.
    horizontal_linear_speed = float(abs(horizontal_linear_speed)) # Only keep magnitude.
    radius_of_curvature = float(abs(radius_of_curvature)) # Only keep magnitude.
    total_turn_degree_deg = float(abs(total_turn_degree_deg)) # Only keep magnitude.
    atom_segment = float(abs(atom_segment))
    # Calculate angular velocity. Angular velocity = horizontal_linear_speed/radius_of_curvature
    angular_velocity = horizontal_linear_speed/radius_of_curvature
    # Split circle into fine polygon.
    atom_angle_rad = atom_segment / radius_of_curvature
    atom_angle_deg = atom_angle_rad / factor_deg2rad # Convert deg to rad.
    # Time interval between sending two consecutive speed specify command.
    delta_t = atom_angle_rad/angular_velocity
    # Calculate Vx and Vy.
    #velocity_x = turn_direction * horizontal_linear_speed * math.sin(atom_angle_rad)
    #velocity_y = horizontal_linear_speed * math.cos(atom_angle_rad)
    # Construct MAVlink message.
    msg_velocity = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, # frame
        0b0000111111000111, # type_mask (only speeds enabled)
        0, 0, 0, # x, y, z positions (not used)
        horizontal_linear_speed, 0, velocity_z, # x, y, z velocity in m/s. x direction is forward.
        0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)
    
    # create the CONDITION_YAW command using command_long_encode()
    msg_yaw = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_CONDITION_YAW, #command
        0, #confirmation
        atom_angle_deg,  # param 1, yaw in degrees
        0,          # param 2, yaw speed deg/s
        turn_direction,  # param 3, direction -1 ccw(subtract), 1 cw(add)
        1, # param 4, if set to 0, yaw is an absolute direction[0-360](0=north, 90=east); if set to 1, yaw is a relative degree to the current yaw direction.
        0, 0, 0)    # param 5 ~ 7 not used
        
    # After taking off, yaw commands are ignored until the first "movement" command has been received. If you need to yaw immediately following takeoff then send a command to "move" to your current position. Make sure send a dummy_movement() before send yaw.
    # Send command to vehicle every delta_t seconds.
    time_temp = time.ctime()
    print('{} - horizontal_linear_speed={} m/s.'.format(time_temp, horizontal_linear_speed))
    print('{} - radius_of_curvature={} m.'.format(time_temp, turn_direction * radius_of_curvature))
    print('{} - atom_segment={} m.'.format(time_temp, atom_segment))
    print('{} - atom_angle_deg={} degree.'.format(time_temp, atom_angle_deg))
    print('{} - delta_t={} s.'.format(time_temp, delta_t))
    print('{} - angular_velocity={} rad/s.'.format(time_temp, angular_velocity))
    print('\n')
    
    remaining_degree_to_turn = total_turn_degree_deg
    while (remaining_degree_to_turn > 0):
        # if remaining_degree_to_turn < atom_angle_deg, generate new msg_yaw.
        if (remaining_degree_to_turn < atom_angle_deg):
            # create the CONDITION_YAW command using command_long_encode()
            msg_yaw = vehicle.message_factory.command_long_encode(
                0, 0,    # target system, target component
                mavutil.mavlink.MAV_CMD_CONDITION_YAW, #command
                0, #confirmation
                remaining_degree_to_turn,  # param 1, yaw in degrees
                0,          # param 2, yaw speed deg/s
                turn_direction,  # param 3, direction -1 ccw(subtract), 1 cw(add)
                1, # param 4, if set to 0, yaw is an absolute direction[0-360](0=north, 90=east); if set to 1, yaw is a relative degree to the current yaw direction.
                0, 0, 0)    # param 5 ~ 7 not used
            # Use shorter delta_t
            delta_t = delta_t * remaining_degree_to_turn / atom_angle_deg
        
        print('{} - Remaining degree to turn = {} degrees'.format(time.ctime(), remaining_degree_to_turn))
        print('{} - Sending mavlink message msg_yaw, atom_angle_deg={}, turn_direction={}.'.format(time.ctime(), atom_angle_deg, turn_direction))
        vehicle.send_mavlink(msg_yaw)
        print('{} - Sending mavlink message msg_velocity, horizontal_linear_speed={}.'.format(time.ctime(), horizontal_linear_speed))
        vehicle.send_mavlink(msg_velocity)
        time.sleep(delta_t)
        remaining_degree_to_turn -= atom_angle_deg
        #get_vehicle_state(vehicle)
        print('\n')

#===================================================

def gps_newpos_1(lat, lon, bearing, distance):
    '''extrapolate latitude/longitude given a heading and distance 
    thanks to http://www.movable-type.co.uk/scripts/latlong.html
    '''
    from math import sin, asin, cos, atan2, radians, degrees

    lat1 = radians(lat)
    lon1 = radians(lon)
    brng = radians(bearing)
    dr = distance/radius_of_earth

    lat2 = asin(sin(lat1)*cos(dr) +
            cos(lat1)*sin(dr)*cos(brng))
    lon2 = lon1 + atan2(sin(brng)*sin(dr)*cos(lat1), 
                cos(dr)-sin(lat1)*sin(lat2))
    return (degrees(lat2), degrees(lon2))

def ROI_OFFSET():
	print ("mm")
	global check_box_flag3       
        global xy_pos, latlon_pos
        global master, no_uavs
        global self_heal
        global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10
        global follower_host_tuple
        global circle_pos_flag

        xoffset = xoffset_entry.get()
        cradius = cradius_entry.get()
        aoffset = aoffset_entry.get()
        salt = salt_entry.get() 
        xoffset = int(xoffset)
        cradius = int(cradius)
        aoffset = int(aoffset)
        salt = int(salt)

	print ("......roi point.....", follower_host_tuple)

        if master == 1:
            lat = vehicle1.location.global_relative_frame.lat
            lon = vehicle1.location.global_relative_frame.lon
            alt = vehicle1.location.global_relative_frame.alt

        if master == 2:
            lat = vehicle2.location.global_relative_frame.lat
            lon = vehicle2.location.global_relative_frame.lon
            alt = vehicle2.location.global_relative_frame.alt

        if master == 3:
            lat = vehicle3.location.global_relative_frame.lat
            lon = vehicle3.location.global_relative_frame.lon
            alt = vehicle3.location.global_relative_frame.alt
                 
        if master == 4:
            lat = vehicle4.location.global_relative_frame.lat
            lon = vehicle4.location.global_relative_frame.lon
            alt = vehicle4.location.global_relative_frame.alt

        if master == 5:
            lat = vehicle5.location.global_relative_frame.lat
            lon = vehicle5.location.global_relative_frame.lon
            alt = vehicle5.location.global_relative_frame.alt

        if circle_pos_flag == True:
                clat = clat_entry.get()
                clon = clon_entry.get()
                clat = float(clat)
                clon = float(clon)
                data_pos = gps_newpos_1(clat, clon, 0, 50)
		clat,clon=data_pos[0],data_pos[1]

        if checkboxvalue_1.get() == 1:
            if circle_pos_flag == True:
                circle_pos_flag = False
		print ("pos", clat,clon)
                alt = vehicle1.location.global_relative_frame.alt
                target = LocationGlobalRelative(clat,clon, alt)
                aggregation_formation(vehicle1,"GUIDED", target)

		
        elif checkboxvalue_2.get() == 1:
            if circle_pos_flag == True:
                circle_pos_flag = False
		print ("pos", clat,clon)
                alt = vehicle2.location.global_relative_frame.alt
                target = LocationGlobalRelative(clat,clon, alt)
                aggregation_formation(vehicle2,"GUIDED", target)

        elif checkboxvalue_3.get() == 1:
            if circle_pos_flag == True:
                circle_pos_flag = False
		print ("pos", clat,clon)
                alt = vehicle3.location.global_relative_frame.alt
                target = LocationGlobalRelative(clat,clon, alt)
                aggregation_formation(vehicle3,"GUIDED", target)

        elif checkboxvalue_4.get() == 1:
            if circle_pos_flag == True:
                circle_pos_flag = False
		print ("pos", clat,clon)
                alt = vehicle4.location.global_relative_frame.alt
                target = LocationGlobalRelative(clat,clon, alt)
                aggregation_formation(vehicle4,"GUIDED", target)

        elif checkboxvalue_5.get() == 1:
            if circle_pos_flag == True:
                circle_pos_flag = False
		print ("pos", clat,clon)
                alt = vehicle5.location.global_relative_frame.alt
                target = LocationGlobalRelative(clat,clon, alt)
                aggregation_formation(vehicle5,"GUIDED", target)
        elif checkboxvalue_6.get() == 1:
            if circle_pos_flag == True:
                circle_pos_flag = False
		print ("pos", clat,clon)
                alt = vehicle6.location.global_relative_frame.alt
                target = LocationGlobalRelative(clat,clon, alt)
                aggregation_formation(vehicle6,"GUIDED", target)
        elif checkboxvalue_7.get() == 1:
            if circle_pos_flag == True:
                circle_pos_flag = False
		print ("pos", clat,clon)
                alt = vehicle7.location.global_relative_frame.alt
                target = LocationGlobalRelative(clat,clon, alt)
                aggregation_formation(vehicle7,"GUIDED", target)
        elif checkboxvalue_8.get() == 1:
            if circle_pos_flag == True:
                circle_pos_flag = False
		print ("pos", clat,clon)
                alt = vehicle8.location.global_relative_frame.alt
                target = LocationGlobalRelative(clat,clon, alt)
                aggregation_formation(vehicle8,"GUIDED", target)
        elif checkboxvalue_9.get() == 1:
            if circle_pos_flag == True:
                circle_pos_flag = False
		print ("pos", clat,clon)
                alt = vehicle9.location.global_relative_frame.alt
                target = LocationGlobalRelative(clat,clon, alt)
                aggregation_formation(vehicle9,"GUIDED", target)
        elif checkboxvalue_10.get() == 1:
            if circle_pos_flag == True:
                circle_pos_flag = False
		print ("pos", clat,clon)
                alt = vehicle10.location.global_relative_frame.alt
                target = LocationGlobalRelative(clat,clon, alt)
                aggregation_formation(vehicle10,"GUIDED", target)



def ROI_heading():
        global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10
        global master
	global check_box_flag3       
  
        global follower_host_tuple
        global circle_pos_flag

        aoffset = aoffset_entry.get()
        salt = salt_entry.get() 

        aoffset = int(aoffset)
        salt = int(salt)
        roi_hd = roi_entry.get()
        roi_hd = int(roi_hd)

        if checkboxvalue_1.get() == 1:
		heading_123(vehicle1,roi_hd)
		
        elif checkboxvalue_2.get() == 1:
		heading_123(vehicle2,roi_hd)

        elif checkboxvalue_3.get() == 1:
		heading_123(vehicle3,roi_hd)

        elif checkboxvalue_4.get() == 1:
		heading_123(vehicle4,roi_hd)
        elif checkboxvalue_5.get() == 1:
		heading_123(vehicle5,roi_hd)
        elif checkboxvalue_6.get() == 1:
		heading_123(vehicle6,roi_hd)
        elif checkboxvalue_7.get() == 1:
		heading_123(vehicle7,roi_hd)
        elif checkboxvalue_8.get() == 1:
		heading_123(vehicle8,roi_hd)
        elif checkboxvalue_9.get() == 1:
		heading_123(vehicle9,roi_hd)
        elif checkboxvalue_10.get() == 1:
		heading_123(vehicle10,roi_hd)
def roi_mode():
        global follower_host_tuple
        global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10
	try:
		ROIradius = ROI_R_entry.get()
		ROIspeed = ROI_S_entry.get()
		ROIradius = int(ROIradius)
		ROIspeed = int(ROIspeed)
	except:
		ROIspeed = 5    # 5 m/s
		ROIradius = 100 # 100 m
	for j, iter_follower in enumerate(follower_host_tuple): 
                if iter_follower == None:
			print ("vehicle is none")
		else:
			time.sleep(0.2)
      			if checkboxvalue_1.get() == 1:
				#horizontal_linear_speed, radius_of_curvature, total_turn_degree_deg, velocity_z, atom_segment
				threading.Thread(target = curvature_flight_body_frame,args=(vehicle1, ROIspeed, -1*ROIradius, 360, 0, 1)).start()	
      			if checkboxvalue_2.get() == 1:
				threading.Thread(target = curvature_flight_body_frame,args=(vehicle2, ROIspeed, -1*ROIradius, 360, 0, 1)).start()	
      			if checkboxvalue_3.get() == 1:
				threading.Thread(target = curvature_flight_body_frame,args=(vehicle3, ROIspeed, -1*ROIradius, 360, 0, 1)).start()	
      			if checkboxvalue_4.get() == 1:
				threading.Thread(target = curvature_flight_body_frame,args=(vehicle4, ROIspeed, -1*ROIradius, 360, 0, 1)).start()	
      			if checkboxvalue_5.get() == 1:
				threading.Thread(target = curvature_flight_body_frame,args=(vehicle5, ROIspeed, -1*ROIradius, 360, 0, 1)).start()	
      			if checkboxvalue_6.get() == 1:
				threading.Thread(target = curvature_flight_body_frame,args=(vehicle6, ROIspeed, -1*ROIradius, 360, 0, 1)).start()	
      			if checkboxvalue_7.get() == 1:
				threading.Thread(target = curvature_flight_body_frame,args=(vehicle7, ROIspeed, -1*ROIradius, 360, 0, 1)).start()	
      			if checkboxvalue_8.get() == 1:
				threading.Thread(target = curvature_flight_body_frame,args=(vehicle8, ROIspeed, -1*ROIradius, 360, 0, 1)).start()	
      			if checkboxvalue_9.get() == 1:
				threading.Thread(target = curvature_flight_body_frame,args=(vehicle9, ROIspeed, -1*ROIradius, 360, 0, 1)).start()	
      			if checkboxvalue_10.get() == 1:
				threading.Thread(target = curvature_flight_body_frame,args=(vehicle10, ROIspeed, -1*ROIradius, 360, 0, 1)).start()		

			time.sleep(0.3)
			print ("thread is ok")

def waypoint():
    global self_heal, master
    global follower_host_tuple

    xoffset = xoffset_entry.get()
    cradius = cradius_entry.get()

    xoffset = int(xoffset)
    cradius = int(cradius)

    print ("all uav waypoint load aggregation_formation")
    if master == 1:		
    	save_mission(vehicle1, export_mission_filename)
    elif master == 2:		
    	save_mission(vehicle2, export_mission_filename)
    elif master == 3:		
    	save_mission(vehicle3, export_mission_filename)
    elif master == 4:		
    	save_mission(vehicle4, export_mission_filename)
    elif master == 5:		
    	save_mission(vehicle5, export_mission_filename)
    elif master == 6:		
    	save_mission(vehicle6, export_mission_filename)

    missionlist = []
    ##a,b,c = (10, 10, 10)
    a,b,c = (xoffset, cradius, 0)
        
    for j, iter_follower in enumerate(follower_host_tuple): 
        if self_heal[j] > 0:    
            print ("lost uav", self_heal[j])
        else:
            print ("else ")
            missionlist = readmission(iter_follower,export_mission_filename,a,b,c,j+1,'even') 
            cmdline(iter_follower,missionlist)
            ##a,b,c = (a+10, b+10, c+20)
            a,b,c = (a+int(xoffset), b+int(cradius), c+5)

            missionlist = []
    missionlist = []    
    time.sleep(0.1)


def auto():
    global check_box_flag3       
    global xy_pos, latlon_pos
    global master, no_uavs
    global self_heal
    global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10,vehicle11,vehicle12,vehicle13
    global vehicle14,vehicle15,vehicle16,vehicle17,vehicle18,vehicle19,vehicle20,vehicle21,vehicle22,vehicle23,vehicle24,vehicle25      
    global follower_host_tuple_G1,follower_host_tuple_G2,follower_host_tuple_G3,follower_host_tuple_G4,follower_host_tuple_G5
    global counter_G1,counter_G2,counter_G3,counter_G4,counter_G5

    global follower_host_tuple
    global circle_pos_flag
    global RTL_all_flag
    xoffset = xoffset_entry.get()
    cradius = cradius_entry.get()
    aoffset = aoffset_entry.get()
    salt = salt_entry.get() 
    xoffset = int(xoffset)
    cradius = int(cradius)
    aoffset = int(aoffset)
    salt = int(salt)
    RTL_all_flag = False
    if checkboxvalue_Group_1.get() == 1:
	Group_1()
	alt_001 = salt
	for i, iter_follower_G1 in enumerate(follower_host_tuple_G1):
	    if iter_follower_G1 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		print ("payload present  uav :", (i+1)) 
		iter_follower_G1.mode = VehicleMode('AUTO')
		time.sleep(0.2)
    if checkboxvalue_Group_2.get() == 1:
	Group_2()
	alt_001 = salt
	for i, iter_follower_G2 in enumerate(follower_host_tuple_G2):
	    if iter_follower_G2 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		print ("payload present  uav :", (i+1)) 
		iter_follower_G2.mode = VehicleMode('AUTO')
		time.sleep(0.2)
    if checkboxvalue_Group_3.get() == 1:
	Group_3()
	alt_001 = salt
	for i, iter_follower_G3 in enumerate(follower_host_tuple_G3):
	    if iter_follower_G3 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		print ("payload present  uav :", (i+1)) 
		iter_follower_G3.mode = VehicleMode('AUTO')
		time.sleep(0.2)
    if checkboxvalue_Group_4.get() == 1:
	Group_4()
	alt_001 = salt
	for i, iter_follower_G4 in enumerate(follower_host_tuple_G4):
	    if iter_follower_G4 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		print ("payload present  uav :", (i+1)) 
		iter_follower_G4.mode = VehicleMode('AUTO')
		time.sleep(0.2)

    if checkboxvalue_Group_all.get() == 1:
	    #for iter_follower in follower_host_tuple: 
	    for i, iter_follower in enumerate(follower_host_tuple):
		if self_heal[i] > 0:
		    print ("slave is lost")
		else:       
		    iter_follower.mode = VehicleMode('AUTO')
		    time.sleep(0.2)

def pause_mission():

    global check_box_flag3       
    global xy_pos, latlon_pos
    global master, no_uavs
    global self_heal
    global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10,vehicle11,vehicle12,vehicle13
    global vehicle14,vehicle15,vehicle16,vehicle17,vehicle18,vehicle19,vehicle20,vehicle21,vehicle22,vehicle23,vehicle24,vehicle25      
    global follower_host_tuple_G1,follower_host_tuple_G2,follower_host_tuple_G3,follower_host_tuple_G4,follower_host_tuple_G5
    global counter_G1,counter_G2,counter_G3,counter_G4,counter_G5

    global follower_host_tuple
    global circle_pos_flag
    global RTL_all_flag
    xoffset = xoffset_entry.get()
    cradius = cradius_entry.get()
    aoffset = aoffset_entry.get()
    salt = salt_entry.get() 
    xoffset = int(xoffset)
    cradius = int(cradius)
    aoffset = int(aoffset)
    salt = int(salt)

    RTL_all_flag = False
    
    if checkboxvalue_Group_1.get() == 1:
	Group_1()
	alt_001 = salt
	for i, iter_follower_G1 in enumerate(follower_host_tuple_G1):
	    if iter_follower_G1 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		print ("payload present  uav :", (i+1)) 
		iter_follower_G1.mode = VehicleMode('LAND')
		time.sleep(0.2)
		iter_follower.mode = VehicleMode('GUIDED')
		time.sleep(0.2)
    if checkboxvalue_Group_2.get() == 1:
	Group_2()
	alt_001 = salt
	for i, iter_follower_G2 in enumerate(follower_host_tuple_G2):
	    if iter_follower_G2 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		print ("payload present  uav :", (i+1)) 
		iter_follower_G2.mode = VehicleMode('LAND')
		time.sleep(0.2)
		iter_follower.mode = VehicleMode('GUIDED')
		time.sleep(0.2)
    if checkboxvalue_Group_3.get() == 1:
	Group_3()
	alt_001 = salt
	for i, iter_follower_G3 in enumerate(follower_host_tuple_G3):
	    if iter_follower_G3 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		print ("payload present  uav :", (i+1)) 
		iter_follower_G3.mode = VehicleMode('LAND')
		time.sleep(0.2)
		iter_follower.mode = VehicleMode('GUIDED')
		time.sleep(0.2)
    if checkboxvalue_Group_4.get() == 1:
	Group_4()
	alt_001 = salt
	for i, iter_follower_G4 in enumerate(follower_host_tuple_G4):
	    if iter_follower_G4 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		print ("payload present  uav :", (i+1)) 
		iter_follower_G4.mode = VehicleMode('LAND')
		time.sleep(0.2)
		iter_follower.mode = VehicleMode('GUIDED')
		time.sleep(0.2)

    if checkboxvalue_Group_all.get() == 1:
	    #for iter_follower in follower_host_tuple: 
	    for i, iter_follower in enumerate(follower_host_tuple):
		if self_heal[i] > 0:
		    print ("slave is lost")
		else: 
		    for i in range(0, 2):      
			    iter_follower.mode = VehicleMode('LAND')
			    time.sleep(0.2)
			    iter_follower.mode = VehicleMode('GUIDED')
			    time.sleep(0.2)


def rtl():
    global check_box_flag3       
    global xy_pos, latlon_pos
    global master, no_uavs
    global self_heal
    global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10,vehicle11,vehicle12,vehicle13
    global vehicle14,vehicle15,vehicle16,vehicle17,vehicle18,vehicle19,vehicle20,vehicle21,vehicle22,vehicle23,vehicle24,vehicle25      
    global follower_host_tuple_G1,follower_host_tuple_G2,follower_host_tuple_G3,follower_host_tuple_G4,follower_host_tuple_G5
    global counter_G1,counter_G2,counter_G3,counter_G4,counter_G5

    global follower_host_tuple
    global circle_pos_flag
    xoffset = xoffset_entry.get()
    cradius = cradius_entry.get()
    aoffset = aoffset_entry.get()
    salt = salt_entry.get() 
    xoffset = int(xoffset)
    cradius = int(cradius)
    aoffset = int(aoffset)
    salt = int(salt)
    global RTL_all_flag
    RTL_all_flag = True
    if checkboxvalue_Group_1.get() == 1:
	Group_1()
	alt_001 = salt
	for i, iter_follower_G1 in enumerate(follower_host_tuple_G1):
	    if iter_follower_G1 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		print ("payload present  uav :", (i+1)) 
		iter_follower_G1.mode = VehicleMode('RTL')
		time.sleep(0.2)
    if checkboxvalue_Group_2.get() == 1:
	Group_2()
	alt_001 = salt
	for i, iter_follower_G2 in enumerate(follower_host_tuple_G2):
	    if iter_follower_G2 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		print ("payload present  uav :", (i+1)) 
		iter_follower_G2.mode = VehicleMode('RTL')
		time.sleep(0.2)
    if checkboxvalue_Group_3.get() == 1:
	Group_3()
	alt_001 = salt
	for i, iter_follower_G3 in enumerate(follower_host_tuple_G3):
	    if iter_follower_G3 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		print ("payload present  uav :", (i+1)) 
		iter_follower_G3.mode = VehicleMode('RTL')
		time.sleep(0.2)
    if checkboxvalue_Group_4.get() == 1:
	Group_4()
	alt_001 = salt
	for i, iter_follower_G4 in enumerate(follower_host_tuple_G4):
	    if iter_follower_G4 == None:
		if check_box_flag3 == True: 
			print ("self heal..to alt change")   
		else:
			alt_001 = alt_001
			#alt_001 = alt_001 + aoffset+10   #....... alt not change during self heal...
	    else: 
		print ("payload present  uav :", (i+1)) 
		iter_follower_G4.mode = VehicleMode('RTL')
		time.sleep(0.2)

    if checkboxvalue_Group_all.get() == 1:
	    #for iter_follower in follower_host_tuple: 
	    for i, iter_follower in enumerate(follower_host_tuple):
		if self_heal[i] > 0:
		    print ("slave is lost")
		else:       
		    iter_follower.mode = VehicleMode('RTL')
		    time.sleep(0.2)

def zoom_plus():
     global ZOOM
     ZOOM = 19
     print ("muthuuuuuuuuuuuuu")



def update_gui():
    global vehicle1, vehicle2, vehicle3, vehicle4,vehicle5,vehicle6,vehicle7,vehicle8,vehicle9,vehicle10,vehicle11,vehicle12,vehicle13
    global vehicle14,vehicle15,vehicle16,vehicle17,vehicle18,vehicle19,vehicle20,vehicle21,vehicle22,vehicle23,vehicle24,vehicle25
    global ip_status 
    

    ##text.place(x =690, y = 632)
    lat1.place(x = 95, y = 40, height=30, width=125)
    longt1.place(x = 220, y = 40, height=30, width=115)
    alt1.place(x = 335, y =40, height=30, width=68)
    airs1.place(x = 489, y = 40, height=30, width=58)
   # sat1.place(x = 455, y = 40, height=30, width=55)
    bat1.place(x = 403, y = 40, height=30, width=85)
    """
    link1.place(x = 595, y = 40, height=30, width=65)
    rssi1.place(x = 520, y = 40, height=30, width=65)
    #mag1.place(x = 725, y = 40, height=30, width=95)
    """
  #  ekf1.place(x = 595, y = 40, height=30, width=90)  
    arm1.place(x = 548, y = 40, height=30, width=60)
    mode1.place(x = 610, y = 40, height=30, width=100)
    """
    s_lat1.place(x = 1030, y = 40, width = 85, height = 30)
    s_lon1.place(x = 1120, y = 40, width = 85, height = 30)
    g_lat1.place(x = 1290, y = 40, width = 85, height = 30)
    g_lon1.place(x = 1380, y = 40, width = 85, height = 30)
    g_alt1.place(x = 1520, y = 40, width = 40, height = 30)
   
    T_lat1.place(x = 1700-10, y = 40, width = 85, height = 30)
    T_lon1.place(x = 1800-25, y = 40, width = 85, height = 30)
    T_score1.place(x = 1900-35, y = 40, width = 40, height = 30)
    """

    
    lat2.place(x = 95, y = 75, height=30, width=125)
    longt2.place(x = 220, y = 75, height=30, width=115)
    alt2.place(x = 335, y =75, height=30, width=68)
    airs2.place(x = 489, y = 75, height=30, width=58)
   # sat2.place(x = 455, y = 75, height=30, width=55)
    bat2.place(x = 403, y = 75, height=30, width=85)
    """
    link2.place(x = 595, y = 75, height=30, width=65)
    rssi2.place(x = 660, y = 75, height=30, width=65)
    #mag2.place(x = 725, y = 75, height=30, width=95)
    """
  #  ekf2.place(x = 595, y = 75, height=30, width=90)  
    arm2.place(x = 548, y = 75, height=30, width=60)
    mode2.place(x = 610, y = 75, height=30, width=100)
    """
    s_lat2.place(x = 1030, y = 75, width = 85, height = 30)
    s_lon2.place(x = 1120, y = 75, width = 85, height = 30)
    g_lat2.place(x = 1290, y = 75, width = 85, height = 30)
    g_lon2.place(x = 1380, y = 75, width = 85, height = 30)
    g_alt2.place(x = 1520, y = 75, width = 40, height = 30)
    
    T_lat2.place(x = 1700-10, y = 75, width = 85, height = 30)
    T_lon2.place(x = 1800-25, y = 75, width = 85, height = 30)
    T_score2.place(x = 1900-35, y = 75, width = 40, height = 30)
    """

    lat3.place(x = 95, y = 110, height=30, width=125)
    longt3.place(x = 220, y = 110, height=30, width=115)
    alt3.place(x = 335, y =110, height=30, width=68)
    airs3.place(x = 489, y = 110, height=30, width=58)
    #sat3.place(x = 455, y = 110, height=30, width=55)
    bat3.place(x = 403, y = 110, height=30, width=85)
    """
    link3.place(x = 595, y = 110, height=30, width=65)
    rssi3.place(x = 660, y = 110, height=30, width=65)
    #mag3.place(x = 725, y = 110, height=30, width=95)
    
    ekf3.place(x = 595, y = 110, height=30, width=90) 
    """ 
    arm3.place(x = 548, y = 110, height=30, width=60)
    
    mode3.place(x = 610, y = 110, height=30, width=100)
    """
    s_lat3.place(x = 1030, y = 110, width = 85, height = 30)
    s_lon3.place(x = 1120, y = 110, width = 85, height = 30)
    g_lat3.place(x = 1290, y = 110, width = 85, height = 30)
    g_lon3.place(x = 1380, y = 110, width = 85, height = 30)
    g_alt3.place(x = 1520, y = 110, width = 40, height = 30)
    
    T_lat3.place(x = 1700-10, y = 110, width = 85, height = 30)
    T_lon3.place(x = 1800-25, y = 110, width = 85, height = 30)
    T_score3.place(x = 1900-35, y = 110, width = 40, height = 30)
    """

    lat4.place(x = 95, y = 145, height=30, width=125)
    longt4.place(x = 220, y = 145, height=30, width=115)
    alt4.place(x = 335, y =145, height=30, width=68)
    
    airs4.place(x = 489, y = 145, height=30, width=58)
    #sat4.place(x = 455, y = 145, height=30, width=55)
    
    bat4.place(x = 403, y = 145, height=30, width=85)
    """
    link4.place(x = 595, y = 145, height=30, width=65)
    rssi4.place(x = 660, y = 145, height=30, width=65)
    #mag4.place(x = 725, y = 145, height=30, width=95)
    
    ekf4.place(x = 595, y = 145, height=30, width=90)  
    """
    arm4.place(x = 548, y = 145, height=30, width=60)
    
    mode4.place(x = 610, y = 145, height=30, width=100)
    """
    s_lat4.place(x = 1030, y = 145, width = 85, height = 30)
    s_lon4.place(x = 1120, y = 145, width = 85, height = 30)
    g_lat4.place(x = 1290, y = 145, width = 85, height = 30)
    g_lon4.place(x = 1380, y = 145, width = 85, height = 30)
    g_alt4.place(x = 1520, y = 145, width = 40, height = 30)
    
    T_lat4.place(x = 1700-10, y = 145, width = 85, height = 30)
    T_lon4.place(x = 1800-25, y = 145, width = 85, height = 30)
    T_score4.place(x = 1900-35, y = 145, width = 40, height = 30)
    """


    lat5.place(x = 95, y = 180, height=30, width=125)
    longt5.place(x = 220, y = 180, height=30, width=115)
    alt5.place(x = 335, y =180, height=30, width=68)
    airs5.place(x = 489, y = 180, height=30, width=58)
    
   # sat5.place(x = 455, y = 180, height=30, width=55)
    bat5.place(x = 403, y = 180, height=30, width=85)
    """
    link5.place(x = 595, y = 180, height=30, width=65)
    rssi5.place(x = 660, y = 180, height=30, width=65)
    #mag5.place(x = 725, y = 180, height=30, width=95)
    ekf5.place(x = 595, y = 180, height=30, width=90)  
    """  
    arm5.place(x = 548, y = 180, height=30, width=60)
    
    mode5.place(x = 610, y = 180, height=30, width=100)
    """
    s_lat5.place(x = 1030, y = 180, width = 85, height = 30)
    s_lon5.place(x = 1120, y = 180, width = 85, height = 30)
    g_lat5.place(x = 1290, y = 180, width = 85, height = 30)
    g_lon5.place(x = 1380, y = 180, width = 85, height = 30)
    g_alt5.place(x = 1520, y = 180, width = 40, height = 30)
    
    T_lat5.place(x = 1700-10, y = 180, width = 85, height = 30)
    T_lon5.place(x = 1800-25, y = 180, width = 85, height = 30)
    T_score5.place(x = 1900-35, y = 180, width = 40, height = 30)
    """


    lat6.place(x = 95, y = 215, height=30, width=125)
    longt6.place(x = 220, y = 215, height=30, width=115)
    alt6.place(x = 335, y =215, height=30, width=68)
    
    airs6.place(x = 489, y = 215, height=30, width=58)
    #sat6.place(x = 455, y = 215, height=30, width=55)
    
    bat6.place(x = 403, y = 215, height=30, width=85)
    """
    link6.place(x = 595, y = 215, height=30, width=65)
    #rssi6.place(x = 660, y = 215, height=30, width=65)
    #mag6.place(x = 725, y = 215, height=30, width=95)
    
    ekf6.place(x = 595, y = 215, height=30, width=90) 
    """ 
    arm6.place(x = 548, y = 215, height=30, width=60)
    
    mode6.place(x = 610, y = 215, height=30, width=100)
    """
    s_lat6.place(x = 1030, y = 215, width = 85, height = 30)
    s_lon6.place(x = 1120, y = 215, width = 85, height = 30)
    g_lat6.place(x = 1290, y = 215, width = 85, height = 30)
    g_lon6.place(x = 1380, y = 215, width = 85, height = 30)
    g_alt6.place(x = 1520, y = 215, width = 40, height = 30)
    
    T_lat6.place(x = 1700-10, y = 215, width = 85, height = 30)
    T_lon6.place(x = 1800-25, y = 215, width = 85, height = 30)
    T_score6.place(x = 1900-35, y = 215, width = 40, height = 30)
    """

    lat7.place(x = 95, y = 250, height=30, width=125)
    longt7.place(x = 220, y = 250, height=30, width=115)
    alt7.place(x = 335, y =250, height=30, width=68)
    
    airs7.place(x = 489, y = 250, height=30, width=58)
    #sat7.place(x = 455, y = 250, height=30, width=55)
    
    bat7.place(x = 403, y = 250, height=30, width=85)
    """
    link7.place(x = 595, y = 250, height=30, width=65)
    rssi7.place(x = 660, y = 250, height=30, width=65)
    #mag7.place(x = 725, y = 250, height=30, width=95)
    
    ekf7.place(x = 595, y = 250, height=30, width=90)  
    """
    arm7.place(x = 548, y = 250, height=30, width=60)
    
    mode7.place(x = 610, y = 250, height=30, width=100)
    """
    s_lat7.place(x = 1030, y = 250, width = 85, height = 30)
    s_lon7.place(x = 1120, y = 250, width = 85, height = 30)
    g_lat7.place(x = 1290, y = 250, width = 85, height = 30)
    g_lon7.place(x = 1380, y = 250, width = 85, height = 30)
    g_alt7.place(x = 1520, y = 250, width = 40, height = 30)
    
    T_lat7.place(x = 1700-10, y = 250, width = 85, height = 30)
    T_lon7.place(x = 1800-25, y = 250, width = 85, height = 30)
    T_score7.place(x = 1900-35, y = 250, width = 40, height = 30)
    """

    lat8.place(x = 95, y = 285, height=30, width=125)
    longt8.place(x = 220, y = 285, height=30, width=115)
    alt8.place(x = 335, y =285, height=30, width=68)
    
    airs8.place(x = 489, y = 285, height=30, width=58)
    #sat8.place(x = 455, y = 285, height=30, width=55)
    
    bat8.place(x = 403, y = 285, height=30, width=85)
    """
    link8.place(x = 595, y = 285, height=30, width=65)
    rssi8.place(x = 660, y = 285, height=30, width=65)
    #mag8.place(x = 725, y = 285, height=30, width=95)
    
    ekf8.place(x = 595, y = 285, height=30, width=90)
    """   
    arm8.place(x = 548, y = 285, height=30, width=60)
    
    mode8.place(x = 610, y = 285, height=30, width=100)
    """
    s_lat8.place(x = 1030, y = 285, width = 85, height = 30)
    s_lon8.place(x = 1120, y = 285, width = 85, height = 30)
    g_lat8.place(x = 1290, y = 285, width = 85, height = 30)
    g_lon8.place(x = 1380, y = 285, width = 85, height = 30)
    g_alt8.place(x = 1520, y = 285, width = 40, height = 30)
    
    T_lat8.place(x = 1700-10, y = 285, width = 85, height = 30)
    T_lon8.place(x = 1800-25, y = 285, width = 85, height = 30)
    T_score8.place(x = 1900-35, y = 285, width = 40, height = 30)
    """

    lat9.place(x = 95, y = 320, height=30, width=125)
    longt9.place(x = 220, y = 320, height=30, width=115)
    alt9.place(x = 335, y =320, height=30, width=68)
       
    airs9.place(x = 489, y = 320, height=30, width=58)
    #sat9.place(x = 455, y = 320, height=30, width=55)
    bat9.place(x = 403, y = 320, height=30, width=85)
    """
    link9.place(x = 595, y = 320, height=30, width=65)
    rssi9.place(x = 660, y = 320, height=30, width=65)
    mag9.place(x = 725, y = 320, height=30, width=95)
    
    ekf9.place(x = 595, y = 320, height=30, width=90)   
    """
    arm9.place(x = 548, y = 320, height=30, width=60)
    
    mode9.place(x = 610, y = 320, height=30, width=100)
    """
    s_lat9.place(x = 1030, y = 320, width = 85, height = 30)
    s_lon9.place(x = 1120, y = 320, width = 85, height = 30)
    g_lat9.place(x = 1290, y = 320, width = 85, height = 30)
    g_lon9.place(x = 1380, y = 320, width = 85, height = 30)
    g_alt9.place(x = 1520, y = 320, width = 40, height = 30)
    
    T_lat9.place(x = 1700-10, y = 320, width = 85, height = 30)
    T_lon9.place(x = 1800-25, y = 320, width = 85, height = 30)
    T_score9.place(x = 1900-35, y = 320, width = 40, height = 30)
    """

    lat10.place(x = 95, y = 355, height=30, width=125)
    longt10.place(x = 220, y = 355, height=30, width=115)
    alt10.place(x = 335, y =355, height=30, width=68)
    
    airs10.place(x = 489, y = 355, height=30, width=58)
    #sat10.place(x = 455, y = 355, height=30, width=55)
    bat10.place(x = 403, y = 355, height=30, width=85)
    """
    link10.place(x = 595, y = 355, height=30, width=65)
    rssi10.place(x = 660, y = 355, height=30, width=65)
    ##mag10.place(x = 725, y = 355, height=30, width=95)
    
    ekf10.place(x = 595, y = 355, height=30, width=90) 
    """
    arm10.place(x = 548, y = 355, height=30, width=60)
    
    mode10.place(x = 610, y = 355, height=30, width=100)
    """
    s_lat10.place(x = 1030, y = 355, width = 85, height = 30)
    s_lon10.place(x = 1120, y = 355, width = 85, height = 30)
    g_lat10.place(x = 1290, y = 355, width = 85, height = 30)
    g_lon10.place(x = 1380, y = 355, width = 85, height = 30)
    g_alt10.place(x = 1520, y = 355, width = 40, height = 30)
    
    T_lat10.place(x = 1700-10, y = 355, width = 85, height = 30)
    T_lon10.place(x = 1800-25, y = 355, width = 85, height = 30)
    T_score10.place(x = 1900-35, y = 355, width = 40, height = 30)
    """

    lat11.place(x = 95, y = 390, height=30, width=125)
    longt11.place(x = 220, y = 390, height=30, width=115)
    alt11.place(x = 335, y =390, height=30, width=68)
    airs11.place(x = 489, y = 390, height=30, width=58)
    #sat11.place(x = 455, y = 390, height=30, width=55)
    bat11.place(x = 403, y = 390, height=30, width=85)
    """
    link11.place(x = 595, y = 390, height=30, width=65)
    rssi11.place(x = 660, y = 390, height=30, width=65)
    ##mag11.place(x = 725, y = 390, height=30, width=95)
    
    ekf11.place(x = 595, y = 390, height=30, width=90)
    """
    arm11.place(x = 548, y = 390, height=30, width=60)
    
    
    mode11.place(x = 610, y = 390, height=30, width=100)
    """
    s_lat11.place(x = 1030, y = 390, width = 85, height = 30)
    s_lon11.place(x = 1120, y = 390, width = 85, height = 30)
    g_lat11.place(x = 1290, y = 390, width = 85, height = 30)
    g_lon11.place(x = 1380, y = 390, width = 85, height = 30)
    g_alt11.place(x = 1520, y = 390, width = 40, height = 30)
    
    T_lat11.place(x = 1700-10, y = 390, width = 85, height = 30)
    T_lon11.place(x = 1800-25, y = 390, width = 85, height = 30)
    T_score11.place(x = 1900-35, y = 390, width = 40, height = 30)
    """

    lat12.place(x = 95, y = 425, height=30, width=125)
    longt12.place(x = 220, y = 425, height=30, width=115)
    alt12.place(x = 335, y =425, height=30, width=68)
    airs12.place(x = 489, y = 425, height=30, width=58)
 #   sat12.place(x = 455, y = 425, height=30, width=55)
    bat12.place(x = 403, y = 425, height=30, width=85)
    """
    link12.place(x = 595, y = 425, height=30, width=65)
    rssi12.place(x = 660, y = 425, height=30, width=65)
    ##mag12.place(x = 725, y = 425, height=30, width=95)
    
    ekf12.place(x = 595, y = 425, height=30, width=90)   
    """
    arm12.place(x = 548, y = 425, height=30, width=60)
    
    mode12.place(x = 610, y = 425, height=30, width=100)
    """
    s_lat12.place(x = 1030, y = 425, width = 85, height = 30)
    s_lon12.place(x = 1120, y = 425, width = 85, height = 30)
    g_lat12.place(x = 1290, y = 425, width = 85, height = 30)
    g_lon12.place(x = 1380, y = 425, width = 85, height = 30)
    g_alt12.place(x = 1520, y = 425, width = 40, height = 30)
    
    T_lat12.place(x = 1700-10, y = 425, width = 85, height = 30)
    T_lon12.place(x = 1800-25, y = 425, width = 85, height = 30)
    T_score12.place(x = 1900-35, y = 425, width = 40, height = 30)
    """

    lat13.place(x = 95, y = 460, height=30, width=125)
    longt13.place(x = 220, y = 460, height=30, width=115)
    alt13.place(x = 335, y =460, height=30, width=68)
    airs13.place(x = 489, y = 460, height=30, width=58)
 #   sat13.place(x = 455, y = 460, height=30, width=55)
    bat13.place(x = 403, y = 460, height=30, width=85)
    """
    link13.place(x = 595, y = 460, height=30, width=65)
    rssi13.place(x = 660, y = 460, height=30, width=65)
    ##mag13.place(x = 725, y = 460, height=30, width=95)
    
    ekf13.place(x = 595, y = 460, height=30, width=90) 
    """ 
    arm13.place(x = 548, y = 460, height=30, width=60)
    
    mode13.place(x = 610, y = 460, height=30, width=100)
    """
    s_lat13.place(x = 1030, y = 460, width = 85, height = 30)
    s_lon13.place(x = 1120, y = 460, width = 85, height = 30)
    g_lat13.place(x = 1290, y = 460, width = 85, height = 30)
    g_lon13.place(x = 1380, y = 460, width = 85, height = 30)
    g_alt13.place(x = 1520, y = 460, width = 40, height = 30)
    
    T_lat13.place(x = 1700-10, y = 460, width = 85, height = 30)
    T_lon13.place(x = 1800-25, y = 460, width = 85, height = 30)
    T_score13.place(x = 1900-35, y = 460, width = 40, height = 30)
    """


    lat14.place(x = 95, y = 495, height=30, width=125)
    longt14.place(x = 220, y = 495, height=30, width=115)
    alt14.place(x = 335, y =495, height=30, width=68)
    
    airs14.place(x = 489, y = 495, height=30, width=58)
    #sat14.place(x = 455, y = 495, height=30, width=55)
    
    bat14.place(x = 403, y = 495, height=30, width=85)
    """
    link14.place(x = 595, y = 495, height=30, width=65)
    rssi14.place(x = 660, y = 495, height=30, width=65)
    ##mag14.place(x = 725, y = 495, height=30, width=95)
    
    ekf14.place(x = 595, y = 495, height=30, width=90) 
    """ 
    arm14.place(x = 548, y = 495, height=30, width=60)
    
    mode14.place(x = 610, y = 495, height=30, width=100)
    """
    s_lat14.place(x = 1030, y = 495, width = 85, height = 30)
    s_lon14.place(x = 1120, y = 495, width = 85, height = 30)
    g_lat14.place(x = 1290, y = 495, width = 85, height = 30)
    g_lon14.place(x = 1380, y = 495, width = 85, height = 30)
    g_alt14.place(x = 1520, y = 495, width = 40, height = 30)
    
    T_lat14.place(x = 1700-10, y = 495, width = 85, height = 30)
    T_lon14.place(x = 1800-25, y = 495, width = 85, height = 30)
    T_score14.place(x = 1900-35, y = 495, width = 40, height = 30)
    """

    lat15.place(x = 95, y = 530, height=30, width=125)
    longt15.place(x = 220, y = 530, height=30, width=115)
    alt15.place(x = 335, y =530, height=30, width=68)
    
    airs15.place(x = 489, y = 530, height=30, width=58)
    #sat15.place(x = 455, y = 530, height=30, width=55)
    bat15.place(x = 403, y = 530, height=30, width=85)
    """
    link15.place(x = 595, y = 530, height=30, width=65)
    rssi15.place(x = 660, y = 530, height=30, width=65)
    ##mag15.place(x = 725, y = 530, height=30, width=95)
    
    ekf15.place(x = 595, y = 530, height=30, width=90)
    """
    arm15.place(x = 548, y = 530, height=30, width=60)
    
    mode15.place(x = 610, y = 530, height=30, width=100)
    """
    s_lat15.place(x = 1030, y = 530, width = 85, height = 30)
    s_lon15.place(x = 1120, y = 530, width = 85, height = 30)
    g_lat15.place(x = 1290, y = 530, width = 85, height = 30)
    g_lon15.place(x = 1380, y = 530, width = 85, height = 30)
    g_alt15.place(x = 1520, y = 530, width = 40, height = 30)
    
    T_lat15.place(x = 1700-10, y = 530, width = 85, height = 30)
    T_lon15.place(x = 1800-25, y = 530, width = 85, height = 30)
    T_score15.place(x = 1900-35, y = 530, width = 40, height = 30)
    """

    '''
    lat16.place(x = 95, y = 565, height=30, width=125)
    longt16.place(x = 225, y = 565, height=30, width=115)
    alt16.place(x = 330, y =565, height=30, width=68)
    airs16.place(x = 400, y = 565, height=30, width=58)
    sat16.place(x = 455, y = 565, height=30, width=55)
    bat16.place(x = 510, y = 565, height=30, width=85)
    """
    link16.place(x = 595, y = 565, height=30, width=65)
    rssi16.place(x = 660, y = 565, height=30, width=65)
    ##mag16.place(x = 725, y = 565, height=30, width=95)
    """
    ekf16.place(x = 595, y = 565, height=30, width=90)
    arm16.place(x = 660, y = 565, height=30, width=60)
    mode16.place(x = 725, y = 565, height=30, width=100)
    s_lat16.place(x = 1030, y = 565, width = 85, height = 30)
    s_lon16.place(x = 1120, y = 565, width = 85, height = 30)
    g_lat16.place(x = 1290, y = 565, width = 85, height = 30)
    g_lon16.place(x = 1380, y = 565, width = 85, height = 30)
    g_alt16.place(x = 1520, y = 565, width = 40, height = 30)
    """
    T_lat16.place(x = 1700-10, y = 565, width = 85, height = 30)
    T_lon16.place(x = 1800-25, y = 565, width = 85, height = 30)
    T_score16.place(x = 1900-35, y = 565, width = 40, height = 30)
    """



    lat17.place(x = 95, y = 600, height=30, width=125)
    longt17.place(x = 225, y = 600, height=30, width=115)
    alt17.place(x = 330, y =600, height=30, width=68)
    airs17.place(x = 400, y = 600, height=30, width=58)
    sat17.place(x = 455, y = 600, height=30, width=55)
    bat17.place(x = 510, y = 600, height=30, width=85)
    """
    link17.place(x = 595, y = 600, height=30, width=65)
    rssi17.place(x = 660, y = 600, height=30, width=65)
    ##mag17.place(x = 725, y = 600, height=30, width=95)
    """
    ekf17.place(x = 595, y = 600, height=30, width=90) 
    arm17.place(x = 660, y = 600, height=30, width=60)
    
    mode17.place(x = 725, y = 600, height=30, width=100)
    s_lat17.place(x = 1030, y = 600, width = 85, height = 30)
    s_lon17.place(x = 1120, y = 600, width = 85, height = 30)
    g_lat17.place(x = 1290, y = 600, width = 85, height = 30)
    g_lon17.place(x = 1380, y = 600, width = 85, height = 30)
    g_alt17.place(x = 1520, y = 600, width = 40, height = 30)
    """
    T_lat17.place(x = 1700-10, y = 600, width = 85, height = 30)
    T_lon17.place(x = 1800-25, y = 600, width = 85, height = 30)
    T_score17.place(x = 1900-35, y = 600, width = 40, height = 30)
    """

    lat18.place(x = 95, y = 635, height=30, width=125)
    longt18.place(x = 225, y = 635, height=30, width=115)
    alt18.place(x = 330, y =635, height=30, width=68)
    airs18.place(x = 400, y = 635, height=30, width=58)
    sat18.place(x = 455, y = 635, height=30, width=55)
    bat18.place(x = 510, y = 635, height=30, width=85)
    """
    link18.place(x = 595, y = 635, height=30, width=65)
    rssi18.place(x = 660, y = 635, height=30, width=65)
    ##mag18.place(x = 725, y = 635, height=30, width=95)
    """
    ekf18.place(x = 595, y = 635, height=30, width=90)   
    arm18.place(x = 660, y = 635, height=30, width=60)
    mode18.place(x = 725, y = 635, height=30, width=100)
    s_lat18.place(x = 1030, y = 635, width = 85, height = 30)
    s_lon18.place(x = 1120, y = 635, width = 85, height = 30)
    g_lat18.place(x = 1290, y = 635, width = 85, height = 30)
    g_lon18.place(x = 1380, y = 635, width = 85, height = 30)
    g_alt18.place(x = 1520, y = 635, width = 40, height = 30)
    """
    T_lat18.place(x = 1700-10, y = 635, width = 85, height = 30)
    T_lon18.place(x = 1800-25, y = 635, width = 85, height = 30)
    T_score18.place(x = 1900-35, y = 635, width = 40, height = 30)
    """

    lat19.place(x = 95, y = 670, height=30, width=125)
    longt19.place(x = 225, y = 670, height=30, width=115)
    alt19.place(x = 330, y =670, height=30, width=68)
    airs19.place(x = 400, y = 670, height=30, width=58)
    sat19.place(x = 455, y = 670, height=30, width=55)
    bat19.place(x = 510, y = 670, height=30, width=85)
    """
    link19.place(x = 595, y = 670, height=30, width=65)
    rssi19.place(x = 660, y = 670, height=30, width=65)
    ##mag19.place(x = 725, y = 670, height=30, width=95)
    """
    ekf19.place(x = 595, y = 670, height=30, width=90)  
    arm19.place(x = 660, y = 670, height=30, width=60)
    mode19.place(x = 725, y = 670, height=30, width=100)
    s_lat19.place(x = 1030, y = 670, width = 85, height = 30)
    s_lon19.place(x = 1120, y = 670, width = 85, height = 30)
    g_lat19.place(x = 1290, y = 670, width = 85, height = 30)
    g_lon19.place(x = 1380, y = 670, width = 85, height = 30)
    g_alt19.place(x = 1520, y = 670, width = 40, height = 30)
    """
    T_lat19.place(x = 1700-10, y = 670, width = 85, height = 30)
    T_lon19.place(x = 1800-25, y = 670, width = 85, height = 30)
    T_score19.place(x = 1900-35, y = 670, width = 40, height = 30)
    """


    lat20.place(x = 95, y = 705, height=30, width=125)
    longt20.place(x = 225, y = 705, height=30, width=115)
    alt20.place(x = 330, y =705, height=30, width=68)
    airs20.place(x = 400, y = 705, height=30, width=58)
    sat20.place(x = 455, y = 705, height=30, width=55)
    bat20.place(x = 510, y = 705, height=30, width=85)
    """
    link20.place(x = 595, y = 705, height=30, width=65)
    rssi20.place(x = 660, y = 705, height=30, width=65)
    ##mag20.place(x = 725, y = 705, height=30, width=95)
    """
    ekf20.place(x = 595, y = 705, height=30, width=90)  
    arm20.place(x = 660, y = 705, height=30, width=60)
    mode20.place(x = 725, y = 705, height=30, width=100)
    s_lat20.place(x = 1030, y = 705, width = 85, height = 30)
    s_lon20.place(x = 1120, y = 705, width = 85, height = 30)
    g_lat20.place(x = 1290, y = 705, width = 85, height = 30)
    g_lon20.place(x = 1380, y = 705, width = 85, height = 30)
    g_alt20.place(x = 1520, y = 705, width = 40, height = 30)
    """
    T_lat20.place(x = 1700-10, y = 705, width = 85, height = 30)
    T_lon20.place(x = 1800-25, y = 705, width = 85, height = 30)
    T_score20.place(x = 1900-35, y = 705, width = 40, height = 30)
    """

    lat21.place(x = 95, y = 740, height=30, width=125)
    longt21.place(x = 225, y = 740, height=30, width=115)
    alt21.place(x = 330, y =740, height=30, width=68)
    airs21.place(x = 400, y = 740, height=30, width=58)
    sat21.place(x = 455, y = 740, height=30, width=55)
    bat21.place(x = 510, y = 740, height=30, width=85)
    """
    link21.place(x = 595, y = 740, height=30, width=65)
    #rssi21.place(x = 660, y = 740, height=30, width=65)
    ##mag21.place(x = 725, y = 740, height=30, width=95)
    """
    ekf21.place(x = 595, y = 740, height=30, width=90)   
    arm21.place(x = 660, y = 740, height=30, width=60)
    mode21.place(x = 725, y = 740, height=30, width=100)
    s_lat21.place(x = 1030, y = 740, width = 85, height = 30)
    s_lon21.place(x = 1120, y = 740, width = 85, height = 30)
    g_lat21.place(x = 1290, y = 740, width = 85, height = 30)
    g_lon21.place(x = 1380, y = 740, width = 85, height = 30)
    g_alt21.place(x = 1520, y = 740, width = 40, height = 30)
    """
    T_lat21.place(x = 1700-10, y = 740, width = 85, height = 30)
    T_lon21.place(x = 1800-25, y = 740, width = 85, height = 30)
    T_score21.place(x = 1900-35, y = 740, width = 40, height = 30)
    """

    lat22.place(x = 95, y = 775, height=30, width=125)
    longt22.place(x = 225, y = 775, height=30, width=115)
    alt22.place(x = 330, y =775, height=30, width=68)
    airs22.place(x = 400, y = 775, height=30, width=58)
    sat22.place(x = 455, y = 775, height=30, width=55)
    bat22.place(x = 510, y = 775, height=30, width=85)
    """
    link22.place(x = 595, y = 775, height=30, width=65)
    #rssi22.place(x = 660, y = 775, height=30, width=65)
    ##mag22.place(x = 725, y = 775, height=30, width=95)
    """
    ekf22.place(x = 595, y = 775, height=30, width=90)  
    arm22.place(x = 660, y = 775, height=30, width=60)
    mode22.place(x = 725, y = 775, height=30, width=100)
    s_lat22.place(x = 1030, y = 775, width = 85, height = 30)
    s_lon22.place(x = 1120, y = 775, width = 85, height = 30)
    g_lat22.place(x = 1290, y = 775, width = 85, height = 30)
    g_lon22.place(x = 1380, y = 775, width = 85, height = 30)
    g_alt22.place(x = 1520, y = 775, width = 40, height = 30)
    """
    T_lat22.place(x = 1700-10, y = 775, width = 85, height = 30)
    T_lon22.place(x = 1800-25, y = 775, width = 85, height = 30)
    T_score22.place(x = 1900-35, y = 775, width = 40, height = 30)
    """

    lat23.place(x = 95, y = 810, height=30, width=125)
    longt23.place(x = 225, y = 810, height=30, width=115)
    alt23.place(x = 330, y =810, height=30, width=68)
    airs23.place(x = 400, y = 810, height=30, width=58)
    sat23.place(x = 455, y = 810, height=30, width=55)
    bat23.place(x = 510, y = 810, height=30, width=85)
    """
    link23.place(x = 595, y = 810, height=30, width=65)
    #rssi23.place(x = 660, y = 810, height=30, width=65)
    ##mag23.place(x = 725, y = 810, height=30, width=95)
    """
    ekf23.place(x = 595, y = 810, height=30, width=90) 
    arm23.place(x = 660, y = 810, height=30, width=60)
    mode23.place(x = 725, y = 810, height=30, width=100)
    s_lat23.place(x = 1030, y = 810, width = 85, height = 30)
    s_lon23.place(x = 1120, y = 810, width = 85, height = 30)
    g_lat23.place(x = 1290, y = 810, width = 85, height = 30)
    g_lon23.place(x = 1380, y = 810, width = 85, height = 30)
    g_alt23.place(x = 1520, y = 810, width = 40, height = 30)
    """
    T_lat23.place(x = 1700-10, y = 810, width = 85, height = 30)
    T_lon23.place(x = 1800-25, y = 810, width = 85, height = 30)
    T_score23.place(x = 1900-35, y = 810, width = 40, height = 30)
    """

    lat24.place(x = 95, y = 845, height=30, width=125)
    longt24.place(x = 225, y = 845, height=30, width=115)
    alt24.place(x = 330, y =845, height=30, width=68)
    airs24.place(x = 400, y = 845, height=30, width=58)
    sat24.place(x = 455, y = 845, height=30, width=55)
    bat24.place(x = 510, y = 845, height=30, width=85)
    """
    link24.place(x = 595, y = 845, height=30, width=65)
    #rssi24.place(x = 660, y = 845, height=30, width=65)
    ##mag24.place(x = 725, y = 845, height=30, width=95)
    """
    ekf24.place(x = 595, y = 845, height=30, width=90) 
    arm24.place(x = 660, y = 845, height=30, width=60)
    mode24.place(x = 725, y = 845, height=30, width=100)
    s_lat24.place(x = 1030, y = 845, width = 85, height = 30)
    s_lon24.place(x = 1120, y = 845, width = 85, height = 30)
    g_lat24.place(x = 1290, y = 845, width = 85, height = 30)
    g_lon24.place(x = 1380, y = 845, width = 85, height = 30)
    g_alt24.place(x = 1520, y = 845, width = 40, height = 30)
    """
    T_lat24.place(x = 1700-10, y = 845, width = 85, height = 30)
    T_lon24.place(x = 1800-25, y = 845, width = 85, height = 30)
    T_score24.place(x = 1900-35, y = 845, width = 40, height = 30)
    """

    lat25.place(x = 95, y = 880, height=30, width=125)
    longt25.place(x = 225, y = 880, height=30, width=115)
    alt25.place(x = 330, y =880, height=30, width=68)
    airs25.place(x = 400, y = 880, height=30, width=58)
    sat25.place(x = 455, y = 880, height=30, width=55)
    bat25.place(x = 510, y = 880, height=30, width=85)
    """
    link25.place(x = 595, y = 880, height=30, width=65)
    #rssi25.place(x = 660, y = 880, height=30, width=65)
    ##mag25.place(x = 725, y = 880, height=30, width=95)
    """
    ekf25.place(x = 595, y = 880, height=30, width=90)  
    arm25.place(x = 660, y = 880, height=30, width=60)
    mode25.place(x = 725, y = 880, height=30, width=100)
    s_lat25.place(x = 1030, y = 880, width = 85, height = 30)
    s_lon25.place(x = 1120, y = 880, width = 85, height = 30)
    g_lat25.place(x = 1290, y = 880, width = 85, height = 30)
    g_lon25.place(x = 1380, y = 880, width = 85, height = 30)
    g_alt25.place(x = 1520, y = 880, width = 40, height = 30)
    """
    T_lat25.place(x = 1700-10, y = 880, width = 85, height = 30)
    T_lon25.place(x = 1800-25, y = 880, width = 85, height = 30)
    T_score25.place(x = 1900-35, y = 880, width = 40, height = 30)
    """
    '''

    while True:

        time.sleep(0.01)
        try:
            lat1.delete(0, END) 
            longt1.delete(0, END) 
            alt1.delete(0, END) 
            bat1.delete(0, END) 
            #link1.delete(0, END)   #ip link
            mode1.delete(0, END) 
            arm1.delete(0, END)   #ip link
            #rssi1.delete(0, END) 
            airs1.delete(0, END)   #ip link
            #sat1.delete(0, END) 
            #mag1.delete(0, END)   #ip link
            #ekf1.delete(0, END) 


            lat1.insert(0, str(vehicle1.location.global_frame.lat)) 
            longt1.insert(0, str(vehicle1.location.global_frame.lon))
            alt1.insert(0, str((vehicle1.location.global_relative_frame.alt))) 
            bat1.insert(0, str(vehicle1.battery.voltage))
            #link1.insert(0, str(ip_status[0]))
            mode1.insert(0, str(vehicle1.mode.name))
            arm1.insert(0, str(vehicle1.armed))
            #rssi1.insert(0, str(100))
            airs1.insert(0,  str(int(vehicle1.airspeed)))
          #  sat1.insert(0, str(vehicle1.gps_0.satellites_visible))
            #mag1.insert(0, str('Normal'))
         #   ekf1.insert(0, str(vehicle1.ekf_ok))
            battary1 = int(vehicle1.battery.voltage)

            if battary1 <= 46:
                bat1.config(font=helv35, fg = "red")
            else:
                bat1.config(font=helv35, fg = "black")
        
        except:
            pass
        try:
            lat2.delete(0, END) 
            longt2.delete(0, END) 
            alt2.delete(0, END) 
            bat2.delete(0, END)
            #link2.delete(0, END)
            mode2.delete(0, END) 
            arm2.delete(0, END)   #ip link
            #rssi2.delete(0, END) 
            airs2.delete(0, END)   #ip link
            #sat2.delete(0, END) 
            #mag2.delete(0, END)   #ip link
            #ekf2.delete(0, END) 

            lat2.insert(0, str(vehicle2.location.global_frame.lat))
            longt2.insert(0, str(vehicle2.location.global_frame.lon))
            alt2.insert(0, str((vehicle2.location.global_relative_frame.alt))) 
            bat2.insert(0, str(vehicle2.battery.voltage))
            #link2.insert(0,  str(ip_status[1]))
            mode2.insert(0, str(vehicle2.mode.name))
            arm2.insert(0, str(vehicle2.armed))
            #rssi2.insert(0, str(100))
            airs2.insert(0,  str(int(vehicle2.airspeed)))
          #  sat2.insert(0, str(vehicle2.gps_0.satellites_visible))
            #mag2.insert(0, str('Normal'))
         #   ekf2.insert(0, str(vehicle2.ekf_ok))
            battary2 = int(vehicle2.battery.voltage)

            if battary2 <= 46:
                bat2.config(font=helv35, fg = "red")
            else:
                bat2.config(font=helv35, fg = "black")
        except:
            pass

        try:
            lat3.delete(0, END) 
            longt3.delete(0, END) 
            alt3.delete(0, END) 
            bat3.delete(0, END)
            #link3.delete(0, END)
            mode3.delete(0, END) 
            arm3.delete(0, END)   #ip link
            #rssi3.delete(0, END) 
            airs3.delete(0, END)   #ip link
            #sat3.delete(0, END) 
            #mag3.delete(0, END)   #ip link
            #ekf3.delete(0, END) 


            lat3.insert(0, str(vehicle3.location.global_frame.lat))
            longt3.insert(0, str(vehicle3.location.global_frame.lon))
            alt3.insert(0, str((vehicle3.location.global_relative_frame.alt))) 
            bat3.insert(0, str(vehicle3.battery.voltage))
            #link3.insert(0,  str(ip_status[2]))
            mode3.insert(0, str(vehicle3.mode.name))
            arm3.insert(0, str(vehicle3.armed))
            #rssi3.insert(0, str(100))
            airs3.insert(0,  str(int(vehicle3.airspeed)))
            #sat3.insert(0, str(vehicle3.gps_0.satellites_visible))
            #mag3.insert(0, str('Normal'))
            #ekf3.insert(0, str(vehicle3.ekf_ok))
            battary3 = int(vehicle3.battery.voltage)

            if battary3 <= 46:
                bat3.config(font=helv35, fg = "red")
            else:
                bat3.config(font=helv35, fg = "black")
        except:
                pass
        try:

            lat4.delete(0, END) 
            longt4.delete(0, END) 
            alt4.delete(0, END) 
            bat4.delete(0, END)
            #link4.delete(0, END)
            mode4.delete(0, END) 
            arm4.delete(0, END)   #ip link
            #rssi4.delete(0, END) 
            airs4.delete(0, END)   #ip link
          #  sat4.delete(0, END) 
            #mag4.delete(0, END)   #ip link
          #  ekf4.delete(0, END) 

            lat4.insert(0, str(vehicle4.location.global_frame.lat))
            longt4.insert(0, str(vehicle4.location.global_frame.lon))
            alt4.insert(0, str((vehicle4.location.global_relative_frame.alt))) 
            bat4.insert(0, str(vehicle4.battery.voltage))
            #link4.insert(0,  str(ip_status[3]))
            mode4.insert(0, str(vehicle4.mode.name))
            arm4.insert(0, str(vehicle4.armed))
            #rssi4.insert(0, str(100))
            airs4.insert(0,  str(int(vehicle4.airspeed)))
         #   sat4.insert(0, str(vehicle4.gps_0.satellites_visible))
            #mag4.insert(0, str('Normal'))
        #    ekf4.insert(0, str(vehicle4.ekf_ok))
            battary4 = int(vehicle4.battery.voltage)

            if battary4 <= 46:
                bat4.config(font=helv35, fg = "red")
            else:
                bat4.config(font=helv35, fg = "black")
        except:
            pass
        try:

            lat5.delete(0, END) 
            longt5.delete(0, END) 
            alt5.delete(0, END) 
            bat5.delete(0, END) 
            #link5.delete(0, END) 
            mode5.delete(0, END) 
            arm5.delete(0, END)   #ip link
            #rssi5.delete(0, END) 
            airs5.delete(0, END)   #ip link
           # sat5.delete(0, END) 
            #mag5.delete(0, END)   #ip link
            #ekf5.delete(0, END) 

            lat5.insert(0, str(vehicle5.location.global_frame.lat))
            longt5.insert(0, str(vehicle5.location.global_frame.lon))
            alt5.insert(0, str((vehicle5.location.global_relative_frame.alt))) 
            bat5.insert(0, str(vehicle5.battery.voltage))
            #link5.insert(0,  str(ip_status[4]))
            mode5.insert(0, str(vehicle5.mode.name))
            arm5.insert(0, str(vehicle5.armed))
            #rssi5.insert(0, str(100))
            airs5.insert(0,  str(int(vehicle5.airspeed)))
            #sat5.insert(0, str(vehicle5.gps_0.satellites_visible))
            #mag5.insert(0, str('Normal'))
            #ekf5.insert(0, str(vehicle5.ekf_ok))
            battary5 = int(vehicle5.battery.voltage)

            if battary5 <= 46:
                bat5.config(font=helv35, fg = "red")
            else:
                bat5.config(font=helv35, fg = "black")
        except:
            pass
        try:

            lat6.delete(0, END) 
            longt6.delete(0, END) 
            alt6.delete(0, END) 
            bat6.delete(0, END)
            #link6.delete(0, END)
            mode6.delete(0, END) 
            arm6.delete(0, END)   #ip link
            #rssi6.delete(0, END) 
            airs6.delete(0, END)   #ip link
            #sat6.delete(0, END) 
            #mag6.delete(0, END)   #ip link
            #ekf6.delete(0, END) 

            lat6.insert(0, str(vehicle6.location.global_frame.lat))
            longt6.insert(0, str(vehicle6.location.global_frame.lon))
            alt6.insert(0, str((vehicle6.location.global_relative_frame.alt))) 
            bat6.insert(0, str(vehicle6.battery.voltage))
            #link6.insert(0,  str(ip_status[5]))
            mode6.insert(0, str(vehicle6.mode.name))
            arm6.insert(0, str(vehicle6.armed))
            #rssi6.insert(0, str(100))
            airs6.insert(0,  str(int(vehicle6.airspeed)))
            #sat6.insert(0, str(vehicle6.gps_0.satellites_visible))
            #mag6.insert(0, str('Normal'))
            #ekf6.insert(0, str(vehicle6.ekf_ok))
            battary6 = int(vehicle6.battery.voltage)

            if battary6 <= 46:
                bat6.config(font=helv35, fg = "red")
            else:
                bat6.config(font=helv35, fg = "black")
        except:
            pass
        try:
             
            lat7.delete(0, END) 
            longt7.delete(0, END) 
            alt7.delete(0, END) 
            bat7.delete(0, END) 
            #link7.delete(0, END) 
            mode7.delete(0, END) 
            arm7.delete(0, END)   #ip link
            #rssi7.delete(0, END)
            airs7.delete(0, END)   #ip link
           # sat7.delete(0, END) 
            #mag7.delete(0, END)   #ip link
           # ekf7.delete(0, END)  

            lat7.insert(0, str(vehicle7.location.global_frame.lat))
            longt7.insert(0, str(vehicle7.location.global_frame.lon))
            alt7.insert(0, str(vehicle7.location.global_relative_frame.alt)) 
            bat7.insert(0, str(vehicle7.battery.voltage))
            #link7.insert(0,  str(ip_status[6]))
            mode7.insert(0, str(vehicle7.mode.name))
            arm7.insert(0, str(vehicle7.armed))
            #rssi7.insert(0, str(100))
            airs7.insert(0,  str(int(vehicle7.airspeed)))
            #sat7.insert(0, str(vehicle7.gps_0.satellites_visible))
            #mag7.insert(0, str('Normal'))
            #ekf7.insert(0, str(vehicle7.ekf_ok))
            battary7 = int(vehicle7.battery.voltage)
            if battary7 <= 46:
                bat7.config(font=helv35, fg = "red")
            else:
                bat7.config(font=helv35, fg = "black")
        except:
            pass
        try:
             
            lat8.delete(0, END) 
            longt8.delete(0, END) 
            alt8.delete(0, END) 
            bat8.delete(0, END)
            #link8.delete(0, END)
            mode8.delete(0, END) 
            arm8.delete(0, END)   #ip 
            airs8.delete(0, END)   #ip link
            #sat8.delete(0, END) 
            #mag8.delete(0, END)   #ip link
            #ekf8.delete(0, END) #link
            #rssi8.delete(0, END) 

            lat8.insert(0, str(vehicle8.location.global_frame.lat))
            longt8.insert(0, str(vehicle8.location.global_frame.lon))
            alt8.insert(0, str((vehicle8.location.global_relative_frame.alt))) 
            bat8.insert(0, str(vehicle8.battery.voltage))
            #link8.insert(0,  str(ip_status[7]))
            mode8.insert(0, str(vehicle8.mode.name))
            arm8.insert(0, str(vehicle8.armed))
            #rssi8.insert(0, str(100))
            airs8.insert(0,  str(int(vehicle8.airspeed)))
            #sat8.insert(0, str(vehicle8.gps_0.satellites_visible))
            #mag8.insert(0, str('Normal'))
            #ekf8.insert(0, str(vehicle8.ekf_ok))
            battary8 = int(vehicle8.battery.voltage)
            if battary8 <= 46:
                bat8.config(font=helv35, fg = "red")
            else:
                bat8.config(font=helv35, fg = "black")
        except:
            pass
        try:

            lat9.delete(0, END) 
            longt9.delete(0, END) 
            alt9.delete(0, END) 
            bat9.delete(0, END) 
            #link9.delete(0, END) 
            mode9.delete(0, END) 
            arm9.delete(0, END)   #ip link
            #rssi9.delete(0, END) 
            airs9.delete(0, END)   #ip link
            #sat9.delete(0, END) 
            #mag9.delete(0, END)   #ip link
            #ekf9.delete(0, END) 

            lat9.insert(0, str(vehicle9.location.global_frame.lat))
            longt9.insert(0, str(vehicle9.location.global_frame.lon))
            alt9.insert(0, str((vehicle9.location.global_relative_frame.alt))) 
            bat9.insert(0, str(vehicle9.battery.voltage))
            #link9.insert(0,  str(ip_status[8]))
            mode9.insert(0, str(vehicle9.mode.name))
            arm9.insert(0, str(vehicle9.armed))
            #rssi9.insert(0, str(100))
            airs9.insert(0,  str(int(vehicle9.airspeed)))
            #sat9.insert(0, str(vehicle9.gps_0.satellites_visible))
            #mag9.insert(0, str('Normal'))
            #ekf9.insert(0, str(vehicle9.ekf_ok))
            battary9 = int(vehicle9.battery.voltage)

            if battary9 <= 46:
                bat9.config(font=helv35, fg = "red")
            else:
                bat9.config(font=helv35, fg = "black")
        except:
            pass
        try:

            lat10.delete(0, END) 
            longt10.delete(0, END) 
            alt10.delete(0, END) 
            bat10.delete(0, END)
            #link10.delete(0, END)
            mode10.delete(0, END) 
            arm10.delete(0, END)   #ip link
            #rssi10.delete(0, END) 
            airs10.delete(0, END)   #ip link
            #sat10.delete(0, END) 
            ##mag10.delete(0, END)   #ip link
            #ekf10.delete(0, END)

            lat10.insert(0, str(vehicle10.location.global_frame.lat))
            longt10.insert(0, str(vehicle10.location.global_frame.lon))
            alt10.insert(0, str((vehicle10.location.global_relative_frame.alt))) 
            bat10.insert(0, str(vehicle10.battery.voltage))
            #link10.insert(0,  str(ip_status[9]))
            mode10.insert(0, str(vehicle10.mode.name))
            arm10.insert(0, str(vehicle10.armed))
            #rssi10.insert(0, str(100))
            airs10.insert(0,  str(int(vehicle10.airspeed)))
            #sat10.insert(0, str(vehicle10.gps_0.satellites_visible))
            ##mag10.insert(0, str('Normal'))
            #ekf10.insert(0, str(vehicle10.ekf_ok))
            battary10 = int(vehicle10.battery.voltage)

            if battary10 <= 46:
                bat10.config(font=helv35, fg = "red")
            else:
                bat10.config(font=helv35, fg = "black")
        except:
            pass
        try:


            lat11.delete(0, END) 
            longt11.delete(0, END) 
            alt11.delete(0, END) 
            bat11.delete(0, END) 
            #link11.delete(0, END) 
            mode11.delete(0, END) 
            arm11.delete(0, END)   #ip link
            #rssi11.delete(0, END) 
            airs11.delete(0, END)   #ip link
            #sat11.delete(0, END) 
            #mag1.delete(0, END)   #ip link
            #ekf11.delete(0, END) 

            lat11.insert(0, str(vehicle11.location.global_frame.lat))
            longt11.insert(0, str(vehicle11.location.global_frame.lon))
            alt11.insert(0, str((vehicle11.location.global_relative_frame.alt))) 
            bat11.insert(0, str(vehicle11.battery.voltage))
            #link11.insert(0,  str(ip_status[10]))
            mode11.insert(0, str(vehicle11.mode.name))
            arm11.insert(0, str(vehicle11.armed))
            #rssi11.insert(0, str(100))
            airs11.insert(0,  str(int(vehicle11.airspeed)))
            #sat11.insert(0, str(vehicle11.gps_0.satellites_visible))
            ##mag11.insert(0, str('Normal'))
            #ekf11.insert(0, str(vehicle11.ekf_ok))
            battary11 = int(vehicle11.battery.voltage)

            if battary11 <= 46:
                bat11.config(font=helv35, fg = "red")
            else:
                bat11.config(font=helv35, fg = "black")
        except:
            pass
        try:

            lat12.delete(0, END) 
            longt12.delete(0, END) 
            alt12.delete(0, END) 
            bat12.delete(0, END)
            #link12.delete(0, END)
            mode12.delete(0, END) 
            arm12.delete(0, END)   #ip link
            #rssi12.delete(0, END) 
            airs12.delete(0, END)   #ip link
            sat12.delete(0, END) 
            ##mag12.delete(0, END)   #ip link
            ekf12.delete(0, END) 

            lat12.insert(0, str(vehicle12.location.global_frame.lat))
            longt12.insert(0, str(vehicle12.location.global_frame.lon))
            alt12.insert(0, str((vehicle12.location.global_relative_frame.alt))) 
            bat12.insert(0, str(vehicle12.battery.voltage))
            #link12.insert(0,  str(ip_status[11]))
            mode12.insert(0, str(vehicle12.mode.name))
            arm12.insert(0, str(vehicle12.armed))
            #rssi12.insert(0, str(100))
            airs12.insert(0,  str(int(vehicle12.airspeed)))
            sat12.insert(0, str(vehicle12.gps_0.satellites_visible))
            ##mag12.insert(0, str('Normal'))
            ekf12.insert(0, str(vehicle12.ekf_ok))
            battary12 = int(vehicle12.battery.voltage)

            if battary12 <= 46:
                bat12.config(font=helv35, fg = "red")
            else:
                bat12.config(font=helv35, fg = "black")
        except:
            pass
        try:

            lat13.delete(0, END) 
            longt13.delete(0, END) 
            alt13.delete(0, END) 
            bat13.delete(0, END)
            #link13.delete(0, END)
            mode13.delete(0, END) 
            arm13.delete(0, END)   #ip link
            #rssi13.delete(0, END)
            airs13.delete(0, END)   #ip link
            sat13.delete(0, END) 
            ##mag13.delete(0, END)   #ip link
            ekf13.delete(0, END)  

            lat13.insert(0, str(vehicle13.location.global_frame.lat))
            longt13.insert(0, str(vehicle13.location.global_frame.lon))
            alt13.insert(0, str((vehicle13.location.global_relative_frame.alt))) 
            bat13.insert(0, str(vehicle13.battery.voltage))
            #link13.insert(0,  str(ip_status[12]))
            mode13.insert(0, str(vehicle13.mode.name))
            arm13.insert(0, str(vehicle13.armed))
            #rssi13.insert(0, str(100))
            airs13.insert(0,  str(int(vehicle13.airspeed)))
            sat13.insert(0, str(vehicle13.gps_0.satellites_visible))
            ##mag13.insert(0, str('Normal'))
            ekf13.insert(0, str(vehicle13.ekf_ok))
            battary13 = int(vehicle13.battery.voltage)

            if battary13 <= 46:
                bat13.config(font=helv35, fg = "red")
            else:
                bat13.config(font=helv35, fg = "black")
        except:
            pass
        try:

            lat14.delete(0, END) 
            longt14.delete(0, END) 
            alt14.delete(0, END) 
            bat14.delete(0, END)
            #link14.delete(0, END)
            mode14.delete(0, END) 
            arm14.delete(0, END)   #ip link
            #rssi14.delete(0, END) 
            airs14.delete(0, END)   #ip link
            sat14.delete(0, END) 
            ##mag14.delete(0, END)   #ip link
            ekf14.delete(0, END) 

            lat14.insert(0, str(vehicle14.location.global_frame.lat))
            longt14.insert(0, str(vehicle14.location.global_frame.lon))
            alt14.insert(0, str((vehicle14.location.global_relative_frame.alt))) 
            bat14.insert(0, str(vehicle14.battery.voltage))
            #link14.insert(0,  str(ip_status[13]))
            mode14.insert(0, str(vehicle14.mode.name))
            arm14.insert(0, str(vehicle14.armed))
            #rssi14.insert(0, str(100))
            airs14.insert(0,  str(int(vehicle14.airspeed)))
            sat14.insert(0, str(vehicle14.gps_0.satellites_visible))
            ##mag14.insert(0, str('Normal'))
            ekf14.insert(0, str(vehicle14.ekf_ok))
            battary14 = int(vehicle14.battery.voltage)

            if battary14 <= 46:
                bat14.config(font=helv35, fg = "red")
            else:
                bat14.config(font=helv35, fg = "black")
        except:
            pass
        try:

            lat15.delete(0, END) 
            longt15.delete(0, END) 
            alt15.delete(0, END) 
            bat15.delete(0, END) 
            #link15.delete(0, END) 
            mode15.delete(0, END) 
            arm15.delete(0, END)   #ip link
            #rssi15.delete(0, END) 
            airs15.delete(0, END)   #ip link
            sat15.delete(0, END) 
            ##mag15.delete(0, END)   #ip link
            ekf15.delete(0, END) 

            lat15.insert(0, str(vehicle15.location.global_frame.lat))
            longt15.insert(0, str(vehicle15.location.global_frame.lon))
            alt15.insert(0, str((vehicle15.location.global_relative_frame.alt))) 
            bat15.insert(0, str(vehicle15.battery.voltage))
            #link15.insert(0,  str(ip_status[14]))
            mode15.insert(0, str(vehicle15.mode.name))
            arm15.insert(0, str(vehicle15.armed))
            #rssi15.insert(0, str(100))
            airs15.insert(0,  str(int(vehicle15.airspeed)))
            sat15.insert(0, str(vehicle15.gps_0.satellites_visible))
            ##mag15.insert(0, str('Normal'))
            ekf15.insert(0, str(vehicle15.ekf_ok))
            battary15 = int(vehicle15.battery.voltage)
            if battary15 <= 46:
                bat15.config(font=helv35, fg = "red")
            else:
                bat15.config(font=helv35, fg = "black")
        except:
            pass
        '''
        try:

            lat16.delete(0, END) 
            longt16.delete(0, END) 
            alt16.delete(0, END) 
            bat16.delete(0, END)
            #link16.delete(0, END)
            mode16.delete(0, END) 
            arm16.delete(0, END)   #ip link
            #rssi16.delete(0, END) 
            airs16.delete(0, END)   #ip link
            sat16.delete(0, END) 
            ##mag16.delete(0, END)   #ip link
            ekf16.delete(0, END) 

            lat16.insert(0, str(vehicle16.location.global_frame.lat))
            longt16.insert(0, str(vehicle16.location.global_frame.lon))
            alt16.insert(0, str((vehicle16.location.global_relative_frame.alt))) 
            bat16.insert(0, str(vehicle16.battery.voltage))
            #link16.insert(0,  str(ip_status[15]))
            mode16.insert(0, str(vehicle16.mode.name))
            arm16.insert(0, str(vehicle16.armed))
            #rssi16.insert(0, str(100))
            airs16.insert(0,  str(int(vehicle16.airspeed)))
            sat16.insert(0, str(vehicle16.gps_0.satellites_visible))
            ##mag16.insert(0, str('Normal'))
            ekf16.insert(0, str(vehicle16.ekf_ok))
            battary16 = int(vehicle16.battery.voltage)

            if battary16 <= 46:
                bat16.config(font=helv35, fg = "red")
            else:
                bat16.config(font=helv35, fg = "black")
        except:
            pass
        try:

            lat17.delete(0, END) 
            longt17.delete(0, END) 
            alt17.delete(0, END) 
            bat17.delete(0, END) 
            #link17.delete(0, END) 
            mode17.delete(0, END) 
            arm17.delete(0, END)   #ip link
            #rssi17.delete(0, END) 
            airs17.delete(0, END)   #ip link
            sat17.delete(0, END) 
            ##mag17.delete(0, END)   #ip link
            ekf17.delete(0, END) 

            lat17.insert(0, str(vehicle17.location.global_frame.lat))
            longt17.insert(0, str(vehicle17.location.global_frame.lon))
            alt17.insert(0, str((vehicle17.location.global_relative_frame.alt))) 
            bat17.insert(0, str(vehicle17.battery.voltage))
            #link17.insert(0,  str(ip_status[16]))
            mode17.insert(0, str(vehicle17.mode.name))
            arm17.insert(0, str(vehicle17.armed))
            #rssi17.insert(0, str(100))
            airs17.insert(0,  str(int(vehicle17.airspeed)))
            sat17.insert(0, str(vehicle17.gps_0.satellites_visible))
            ##mag17.insert(0, str('Normal'))
            ekf17.insert(0, str(vehicle17.ekf_ok))
            battary17 = int(vehicle17.battery.voltage)

            if battary17 <= 46:
                bat17.config(font=helv35, fg = "red")
            else:
                bat17.config(font=helv35, fg = "black")

        except:
            pass
        try:

            lat18.delete(0, END) 
            longt18.delete(0, END) 
            alt18.delete(0, END) 
            bat18.delete(0, END)
            #link18.delete(0, END)
            mode18.delete(0, END) 
            arm18.delete(0, END)   #ip link
            #rssi18.delete(0, END) 
            airs18.delete(0, END)   #ip link
            sat18.delete(0, END) 
            ##mag18.delete(0, END)   #ip link
            ekf18.delete(0, END) 

            lat18.insert(0, str(vehicle18.location.global_frame.lat))
            longt18.insert(0, str(vehicle18.location.global_frame.lon))
            alt18.insert(0, str((vehicle18.location.global_relative_frame.alt))) 
            bat18.insert(0, str(vehicle18.battery.voltage))
            #link18.insert(0,  str(ip_status[17]))
            mode18.insert(0, str(vehicle18.mode.name))
            arm18.insert(0, str(vehicle18.armed))
            #rssi18.insert(0, str(100))
            airs18.insert(0,  str(int(vehicle18.airspeed)))
            sat18.insert(0, str(vehicle18.gps_0.satellites_visible))
            ##mag18.insert(0, str('Normal'))
            ekf18.insert(0, str(vehicle18.ekf_ok))
            battary18 = int(vehicle18.battery.voltage)

            if battary18 <= 46:
                bat18.config(font=helv35, fg = "red")
            else:
                bat18.config(font=helv35, fg = "black")
        except:
            pass
        try:

            lat19.delete(0, END) 
            longt19.delete(0, END) 
            alt19.delete(0, END) 
            bat19.delete(0, END) 
            #link19.delete(0, END) 
            mode19.delete(0, END) 
            arm19.delete(0, END)   #ip link
            #rssi19.delete(0, END) 
            airs19.delete(0, END)   #ip link
            sat19.delete(0, END) 
            ##mag19.delete(0, END)   #ip link
            ekf19.delete(0, END) 

            lat19.insert(0, str(vehicle19.location.global_frame.lat))
            longt19.insert(0, str(vehicle19.location.global_frame.lon))
            alt19.insert(0, str((vehicle19.location.global_relative_frame.alt))) 
            bat19.insert(0, str(vehicle19.battery.voltage))
            #link19.insert(0,  str(ip_status[18]))
            mode19.insert(0, str(vehicle19.mode.name))
            arm19.insert(0, str(vehicle19.armed))
            #rssi19.insert(0, str(100))
            airs19.insert(0,  str(int(vehicle19.airspeed)))
            sat19.insert(0, str(vehicle19.gps_0.satellites_visible))
            ##mag19.insert(0, str('Normal'))
            ekf19.insert(0, str(vehicle19.ekf_ok))
            battary19 = int(vehicle19.battery.voltage)

            if battary19 <= 46:
                bat19.config(font=helv35, fg = "red")
            else:
                bat19.config(font=helv35, fg = "black")
        except:
            pass
        try:

            lat20.delete(0, END) 
            longt20.delete(0, END) 
            alt20.delete(0, END) 
            bat20.delete(0, END)
            #link20.delete(0, END)
            mode20.delete(0, END) 
            arm20.delete(0, END)   #ip link
            #rssi20.delete(0, END) 
            airs20.delete(0, END)   #ip link
            sat20.delete(0, END) 
            ##mag20.delete(0, END)   #ip link
            ekf20.delete(0, END) 

            lat20.insert(0, str(vehicle20.location.global_frame.lat))
            longt20.insert(0, str(vehicle20.location.global_frame.lon))
            alt20.insert(0, str((vehicle20.location.global_relative_frame.alt))) 
            bat20.insert(0, str(vehicle20.battery.voltage))
            #link20.insert(0,  str(ip_status[19]))
            mode20.insert(0, str(vehicle20.mode.name))
            arm20.insert(0, str(vehicle20.armed))
            #rssi20.insert(0, str(100))
            airs20.insert(0,  str(int(vehicle20.airspeed)))
            sat20.insert(0, str(vehicle20.gps_0.satellites_visible))
            ##mag20.insert(0, str('Normal'))
            ekf20.insert(0, str(vehicle20.ekf_ok))
            battary20 = int(vehicle20.battery.voltage)

            if battary20 <= 46:
                bat20.config(font=helv35, fg = "red")
            else:
                bat20.config(font=helv35, fg = "black")
        except:

            pass

        try:

            lat21.delete(0, END) 
            longt21.delete(0, END) 
            alt21.delete(0, END) 
            bat21.delete(0, END)
            #link21.delete(0, END)
            mode21.delete(0, END) 
            arm21.delete(0, END)   #ip link
            #rssi21.delete(0, END) 
            airs21.delete(0, END)   #ip link
            sat21.delete(0, END) 
            ##mag21.delete(0, END)   #ip link
            ekf21.delete(0, END) 

            lat21.insert(0, str(vehicle21.location.global_frame.lat))
            longt21.insert(0, str(vehicle21.location.global_frame.lon))
            alt21.insert(0, str((vehicle21.location.global_relative_frame.alt))) 
            bat21.insert(0, str(vehicle21.battery.voltage))
            #link21.insert(0,  str(ip_status[15]))
            mode21.insert(0, str(vehicle21.mode.name))
            arm21.insert(0, str(vehicle21.armed))
            #rssi21.insert(0, str(100))
            airs21.insert(0,  str(int(vehicle21.airspeed)))
            sat21.insert(0, str(vehicle21.gps_0.satellites_visible))
            ##mag21.insert(0, str('Normal'))
            ekf21.insert(0, str(vehicle21.ekf_ok))
            battary21 = int(vehicle21.battery.voltage)

            if battary21 <= 46:
                bat21.config(font=helv35, fg = "red")
            else:
                bat21.config(font=helv35, fg = "black")


        except:
            pass
        try:

            lat22.delete(0, END) 
            longt22.delete(0, END) 
            alt22.delete(0, END) 
            bat22.delete(0, END) 
            #link22.delete(0, END) 
            mode22.delete(0, END) 
            arm22.delete(0, END)   #ip link
            #rssi22.delete(0, END) 
            airs22.delete(0, END)   #ip link
            sat22.delete(0, END) 
            ##mag22.delete(0, END)   #ip link
            ekf22.delete(0, END) 

            lat22.insert(0, str(vehicle22.location.global_frame.lat))
            longt22.insert(0, str(vehicle22.location.global_frame.lon))
            alt22.insert(0, str((vehicle22.location.global_relative_frame.alt))) 
            bat22.insert(0, str(vehicle22.battery.voltage))
            #link22.insert(0,  str(ip_status[16]))
            mode22.insert(0, str(vehicle22.mode.name))
            arm22.insert(0, str(vehicle22.armed))
            #rssi22.insert(0, str(100))
            airs22.insert(0,  str(int(vehicle22.airspeed)))
            sat22.insert(0, str(vehicle22.gps_0.satellites_visible))
            ##mag22.insert(0, str('Normal'))
            ekf22.insert(0, str(vehicle22.ekf_ok))
            battary22 = int(vehicle22.battery.voltage)

            if battary22 <= 46:
                bat22.config(font=helv35, fg = "red")
            else:
                bat22.config(font=helv35, fg = "black")
        except:
            pass
        try:

            lat23.delete(0, END) 
            longt23.delete(0, END) 
            alt23.delete(0, END) 
            bat23.delete(0, END)
            #link23.delete(0, END)
            mode23.delete(0, END) 
            arm23.delete(0, END)   #ip link
            #rssi23.delete(0, END) 
            airs23.delete(0, END)   #ip link
            sat23.delete(0, END) 
            ##mag23.delete(0, END)   #ip link
            ekf23.delete(0, END) 

            lat23.insert(0, str(vehicle23.location.global_frame.lat))
            longt23.insert(0, str(vehicle23.location.global_frame.lon))
            alt23.insert(0, str((vehicle23.location.global_relative_frame.alt))) 
            bat23.insert(0, str(vehicle23.battery.voltage))
            #link23.insert(0,  str(ip_status[17]))
            mode23.insert(0, str(vehicle23.mode.name))
            arm23.insert(0, str(vehicle23.armed))
            #rssi23.insert(0, str(100))
            airs23.insert(0,  str(int(vehicle23.airspeed)))
            sat23.insert(0, str(vehicle23.gps_0.satellites_visible))
            ##mag23.insert(0, str('Normal'))
            ekf23.insert(0, str(vehicle23.ekf_ok))
            battary23 = int(vehicle23.battery.voltage)

            if battary23 <= 46:
                bat23.config(font=helv35, fg = "red")
            else:
                bat23.config(font=helv35, fg = "black")
        except:
            pass
        try:

            lat24.delete(0, END) 
            longt24.delete(0, END) 
            alt24.delete(0, END) 
            bat24.delete(0, END) 
            #link24.delete(0, END) 
            mode24.delete(0, END) 
            arm24.delete(0, END)   #ip link
            #rssi24.delete(0, END) 
            airs24.delete(0, END)   #ip link
            sat24.delete(0, END) 
            ##mag24.delete(0, END)   #ip link
            ekf24.delete(0, END) 

            lat24.insert(0, str(vehicle24.location.global_frame.lat))
            longt24.insert(0, str(vehicle24.location.global_frame.lon))
            alt24.insert(0, str((vehicle24.location.global_relative_frame.alt))) 
            bat24.insert(0, str(vehicle24.battery.voltage))
            #link24.insert(0,  str(ip_status[18]))
            mode24.insert(0, str(vehicle24.mode.name))
            arm24.insert(0, str(vehicle24.armed))
            #rssi24.insert(0, str(100))
            airs24.insert(0,  str(int(vehicle24.airspeed)))
            sat24.insert(0, str(vehicle24.gps_0.satellites_visible))
            ##mag24.insert(0, str('Normal'))
            ekf24.insert(0, str(vehicle24.ekf_ok))
            battary24 = int(vehicle24.battery.voltage)

            if battary24 <= 46:
                bat24.config(font=helv35, fg = "red")
            else:
                bat24.config(font=helv35, fg = "black")
        except:
            pass
        try:

            lat25.delete(0, END) 
            longt25.delete(0, END) 
            alt25.delete(0, END) 
            bat25.delete(0, END)
            #link25.delete(0, END)
            mode25.delete(0, END) 
            arm25.delete(0, END)   #ip link
            #rssi25.delete(0, END)
            airs25.delete(0, END)   #ip link
            sat25.delete(0, END) 
            ##mag25.delete(0, END)   #ip link
            ekf25.delete(0, END)  

            lat25.insert(0, str(vehicle25.location.global_frame.lat))
            longt25.insert(0, str(vehicle25.location.global_frame.lon))
            alt25.insert(0, str((vehicle25.location.global_relative_frame.alt))) 
            bat25.insert(0, str(vehicle25.battery.voltage))
            #link25.insert(0,  str(ip_status[19]))
            mode25.insert(0, str(vehicle25.mode.name))
            arm25.insert(0, str(vehicle25.armed))
            #rssi25.insert(0, str(100))
            airs25.insert(0,  str(int(vehicle25.airspeed)))
            sat25.insert(0, str(vehicle25.gps_0.satellites_visible))
            ##mag25.insert(0, str('Normal'))
            ekf25.insert(0, str(vehicle25.ekf_ok))
            battary25 = int(vehicle25.battery.voltage)

            if battary25 <= 46:
                bat25.config(font=helv35, fg = "red")
            else:
                bat25.config(font=helv35, fg = "black")
        except:

            pass
       '''
'''
def show_frame():       
    img0 = Image.open('Dums.jpg')
    img0 = img0.resize((400-45-20-50-20,110), Image.ANTIALIAS)
    img = cv2.cvtColor(np.asarray(img0), cv2.COLOR_RGB2BGR)  
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img) 
    lmain.imgtk = imgtk
    #cv2.imshow("Frame", img)
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


def setCheckButtonText1():
    global check_box_flag1
    if varCheckButton1.get():
        ##varLabel.set("checked")
        print ("checked1")
        check_box_flag1 = True
        message01 = str('manual_mode') + ',' + str(0)
	sent = sock.sendto(message01, server_address)
    else:
        ##varLabel.set("un-checked") 
        print ("un-checked1")
        check_box_flag1 = False
'''

def setCheckButtonText2():
    global search_no_time
    global check_box_flag2, control_command, count_123
    global wp_navigation_stop_forward_flag, wp_navigation_stop_return_flag
    if varCheckButton2.get():
        ##varLabel.set("checked")
        print ("checked2")
	count_123 = 0
	print ("...count_123..", count_123)
	time.sleep(0.1)
        check_box_flag2 = True
        control_command = True
	wp_navigation_stop_forward_flag = False
	wp_navigation_stop_return_flag = False
        message01 = str('auto_mode') + ',' + str(0)
	sent = sock.sendto(message01, server_address)
	search_no_time = search_no_time+1

    else:
        ##varLabel.set("un-checked") 
        print ("un-checked2")
        check_box_flag2 = False
        control_command = False
	wp_navigation_stop_forward_flag = True
	wp_navigation_stop_return_flag = True
	time.sleep(5)
	a5 = threading.Thread(target=autonumus_connection_1)
	a5.daemon = True
	a5.start()

def setCheckButtonText3():
    global check_box_flag3
    if varCheckButton3.get():
        ##varLabel.set("checked")
        print ("checked3...self heal")
	time.sleep(0.1)
        check_box_flag3 = True
        time.sleep(1)
        aggr()


    else:
        ##varLabel.set("un-checked") 
        print ("unchecked3...self heal")
        check_box_flag3 = False
        time.sleep(1)
        aggr()

def setCheckButtonText4():
    global check_box_flag4

    if varCheckButton4.get():
        ##varLabel.set("checked")
        print ("checked4...drop all one point")
	time.sleep(0.1)
        check_box_flag4 = True
        time.sleep(1)

        a30 = threading.Thread(target = payload_drop_all_one_point)
        a30.daemon = True
        a30.start() 
    else:
        ##varLabel.set("un-checked") 
        print ("unchecked4...drop all one point")
        check_box_flag4 = False
        time.sleep(1)

def setCheckButtonText5():
    global check_box_flag5

    if varCheckButton5.get():
        ##varLabel.set("checked")
        print ("checked5...drop all one point line formation")
	time.sleep(0.1)
        check_box_flag5 = True
        time.sleep(1)

	t30 = threading.Thread(target=vehicle_collision_moniter)
	t30.daemon = True
	t30.start()

        a31 = threading.Thread(target = payload_drop_vehicle1_moniter)
        a31.daemon = True
        a31.start() 
        a32 = threading.Thread(target = payload_drop_vehicle2_moniter)
        a32.daemon = True
        a32.start() 
        a33 = threading.Thread(target = payload_drop_vehicle3_moniter)
        a33.daemon = True
        a33.start() 
        a34 = threading.Thread(target = payload_drop_vehicle4_moniter)
        a34.daemon = True
        a34.start() 
        a35 = threading.Thread(target = payload_drop_vehicle5_moniter)
        a35.daemon = True
        a35.start() 
        a36 = threading.Thread(target = payload_drop_vehicle6_moniter)
        a36.daemon = True
        a36.start()
    else:
        ##varLabel.set("un-checked") 
        print ("unchecked5...drop all one point line formation")
        check_box_flag5 = False
        time.sleep(1)



# Driver code 
if __name__ == "__main__" : 

    # Create a GUI window 
    root = Tk() 
    root.geometry("1680x1050") 
    # set the name of tkinter GUI window 
    root.title("Dhaksha micro swarm gcs") 
    root.attributes("-fullscreen", True)  # substitute `Tk` for whatever your `Tk()` object is called
   # frame_1 = Frame(height = 920, width = 1670, bd = 3, relief = 'groove').place(x = 7, y = 5)
    frame_1 = Frame(height = 920, width = 970, bd = 3, relief = 'groove').place(x = 7, y = 5)
    frame_2 = Frame(height = 140, width = 1250+65+50+540, bd = 3, relief = 'groove').place(x = 7, y = 930)
   # frame_4 = tk.Frame(root)
   # frame_4.pack(fill=tk.BOTH, expand=True)
    frame_4 = Frame(height = 920, width = 933, bd = 3, relief = 'groove').place(x = 980, y = 5)#right
    
    ##frame_3 = Frame(height = 120, width = 415, bd = 3, relief = 'groove').place(x = 1260, y = 930)
   # frame_3 = Frame(height = 120, width = 300-20, bd = 3, relief = 'groove').place(x = 1260+45+20+50+20, y = 930)//bottom 2nd frame

  ##  frame_4 = Frame(height = 920, width = 300-70, bd = 3, relief = 'groove').place(x = 1682, y = 5)#right

  #  frame_5 = Frame(height = 120, width = 300-70, bd = 3, relief = 'groove').place(x = 1682, y = 930)

    #frame_6 = Frame(height = 50+2, width = 200-70, bd = 3, relief = 'groove').place(x = 1260, y = 940-3)

    ###frame_3 = Frame(height = 910, width = 815, bd = 3, relief = 'groove', font=("Times New Roman", 15)).place(x = 770, y = 5)
    #frame_4 = Frame(height = 50, width = 670, bd = 3, relief = 'groove', font=("Times New Roman", 15)).place(x = 7, y = 692)
    
    lmain = Label()
    lmain.place(x = 1265+45+20+50+20, y = 935)


    # Set the configuration of GUI window 

    ##Button(text = "DHAKSHA SWARM MICRO GCS", command = zoom_plus, height=5, width = 20, font=("Times New Roman", 15)).place(x = 800, y = 360)
    ##Button(text = "-", command = zoom_minus, height=1, width = 1, font=("Times New Roman", 15)).place(x = 1240, y = 550)
    # Create a Weather Gui Application label 
    #headlabel = Label(root, text = "Weather Gui Application", 
    #fg = 'black', bg = 'red') 
    ##contact01 = Label(text = "control_information", font=("Times New Roman", 15)).place(x = 720, y = 0)
    contact01 = Label(text = "control_information", font=("Times New Roman", 15)).place(x = 60, y = 920)
    ###contact02 = Label(text = "location_information", font=("Times New Roman", 15)).place(x = 60, y = 0)
    contact02 = Label(text = "location_information", font=("Times New Roman", 15)).place(x = 60, y = 0)
    ##contact03 = Label(text = "Swarm_control", font=("Times New Roman", 15)).place(x = 60, y = 690)
    data1_ = Label(text = "UAV_01", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 40)
    data2_ = Label(text = "UAV_02", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 75)
    data3_ = Label(text = "UAV_03", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 110)
    data4_ = Label(text = "UAV_04", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 145)
    data5_ = Label(text = "UAV_05", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 180)
    data6_ = Label(text = "UAV_06", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 215)
    data7_ = Label(text = "UAV_07", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 250)
    data8_ = Label(text = "UAV_08", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 285)
    data9_ = Label(text = "UAV_09", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 320)
    data10_ = Label(text = "UAV_10", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 355)
    data11_ = Label(text = "UAV_11", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 390)
    data12_ = Label(text = "UAV_12", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 425)
    data13_ = Label(text = "UAV_13", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 460)
    data14_ = Label(text = "UAV_14", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 495)
    data15_ = Label(text = "UAV_15", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 530)
    '''
    data16_ = Label(text = "UAV_16", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 565)
    data17_ = Label(text = "UAV_17", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 600)
    data18_ = Label(text = "UAV_18", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 635)
    data19_ = Label(text = "UAV_19", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 670)
    data20_ = Label(text = "UAV_20", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 705)
    data21_ = Label(text = "UAV_21", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 740)
    data22_ = Label(text = "UAV_22", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 775)
    data23_ = Label(text = "UAV_23", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 810)
    data24_ = Label(text = "UAV_24", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 845)
    data25_ = Label(text = "UAV_25", font=("Times New Roman", 15), fg = "green").place(x = 15, y= 880)
    '''
    data26_ = Label(text = "LAT", font=("Times New Roman", 15)).place(x = 140, y= 20)
    data27_ = Label(text = "LONG", font=("Times New Roman", 15)).place(x = 245, y= 20)
    data28_ = Label(text = "ALT", font=("Times New Roman", 15)).place(x = 345, y= 20)
    data29_ = Label(text = "AIRS", font=("Times New Roman", 15)).place(x = 490, y= 20)
    '''
    
    data30_ = Label(text = "SAT", font=("Times New Roman", 15)).place(x = 470, y= 20)
    '''
    data31_ = Label(text = "BAT", font=("Times New Roman", 15)).place(x = 420, y= 20)
    '''
    #data32_ = Label(text = "LINK", font=("Times New Roman", 15)).place(x = 605, y= 20)
    #data33_ = Label(text = "RSSI", font=("Times New Roman", 15)).place(x = 670, y= 20)
    #data34_ = Label(text = "MAG", font=("Times New Roman", 15)).place(x = 745, y= 20)
    data35_ = Label(text = "EKF", font=("Times New Roman", 15)).place(x = 605, y= 20)
    '''   
    data36_ = Label(text = "ARM", font=("Times New Roman", 15)).place(x = 555, y= 20)
    data37_ = Label(text = "MODE", font=("Times New Roman", 15)).place(x = 625, y= 20)
    data38_ = Label(text = "SELECT", font=("Times New Roman", 15)).place(x = 740, y= 20)
    data39_ = Label(text = "SET", font=("Times New Roman", 15)).place(x = 860, y= 20)
    '''
    
    data39_ = Label(text = "CAM", font=("Times New Roman", 15)).place(x = 1000, y= 20)
    data39_ = Label(text = "SLAT", font=("Times New Roman", 15)).place(x = 1030+20, y= 20)
    data39_ = Label(text = "SLON", font=("Times New Roman", 15)).place(x = 1120+10, y= 20)
    data39_ = Label(text = "RC", font=("Times New Roman", 15)).place(x = 1250, y= 20)
    data39_ = Label(text = "GLAT", font=("Times New Roman", 15)).place(x = 1300, y= 20)
    data39_ = Label(text = "GLON", font=("Times New Roman", 15)).place(x = 1385, y= 20)
    data39_ = Label(text = "GALT", font=("Times New Roman", 15)).place(x = 1510, y= 20)
    data39_ = Label(text = "LOAD", font=("Times New Roman", 15)).place(x = 1610, y= 20)
    """
    data39_ = Label(text = "TLAT", font=("Times New Roman", 15)).place(x = 1700, y= 20)
    data39_ = Label(text = "TLON", font=("Times New Roman", 15)).place(x = 1800-10, y= 20)
    data39_ = Label(text = "TALT", font=("Times New Roman", 15)).place(x = 1900-35-10, y= 20)

    """
    '''
    data39_ = Label(text = "PLOT", font=("Times New Roman", 12)).place(x = 1650+80-10, y= 2)
    '''
    data39_ = Label(text = "No", font=("Times New Roman", 12)).place(x = 1700-10-5, y= 20)
    data39_ = Label(text = "G1", font=("Times New Roman", 12)).place(x = 1700+40-10-5, y= 20)
    data39_ = Label(text = "G2", font=("Times New Roman", 12)).place(x = 1700+80-10-5, y= 20)
    data39_ = Label(text = "G3", font=("Times New Roman", 12)).place(x = 1700+120-10-10, y= 20)
    data39_ = Label(text = "G4", font=("Times New Roman", 12)).place(x = 1700+160-10-15, y= 20)
    data39_ = Label(text = "G5", font=("Times New Roman", 12)).place(x = 1700+200-10-20, y= 20)


    data39_ = Label(text = "Group_all", font=("Times New Roman", 12)).place(x = 1700+20, y= 880+60)
    data39_ = Label(text = "Group1", font=("Times New Roman", 12)).place(x = 1700+20+35, y= 880+80)
    data39_ = Label(text = "Group2", font=("Times New Roman", 12)).place(x = 1700+20+35, y= 880+100)
    data39_ = Label(text = "Group3", font=("Times New Roman", 12)).place(x = 1700+20+35, y= 880+120)
    data39_ = Label(text = "Group4", font=("Times New Roman", 12)).place(x = 1700+20+35, y= 880+140)


    data39_ = Label(text = "1", font=("Times New Roman", 12)).place(x = 1690, y= 40)
    data39_ = Label(text = "2", font=("Times New Roman", 12)).place(x = 1690, y= 75)
    data39_ = Label(text = "3", font=("Times New Roman", 12)).place(x = 1690, y= 110)
    data39_ = Label(text = "4", font=("Times New Roman", 12)).place(x = 1690, y= 145)
    data39_ = Label(text = "5", font=("Times New Roman", 12)).place(x = 1690, y= 180)
    data39_ = Label(text = "6", font=("Times New Roman", 12)).place(x = 1690, y= 215)
    data39_ = Label(text = "7", font=("Times New Roman", 12)).place(x = 1690, y= 250)
    data39_ = Label(text = "8", font=("Times New Roman", 12)).place(x = 1690, y= 285)
    data39_ = Label(text = "9", font=("Times New Roman", 12)).place(x = 1690, y= 320)
    data39_ = Label(text = "10", font=("Times New Roman", 12)).place(x = 1690, y= 355)
    data39_ = Label(text = "11", font=("Times New Roman", 12)).place(x = 1690, y= 390)
    data39_ = Label(text = "12", font=("Times New Roman", 12)).place(x = 1690, y= 425)
    data39_ = Label(text = "13", font=("Times New Roman", 12)).place(x = 1690, y= 460)
    data39_ = Label(text = "14", font=("Times New Roman", 12)).place(x = 1690, y= 495)
    data39_ = Label(text = "15", font=("Times New Roman", 12)).place(x = 1690, y= 530)
    data39_ = Label(text = "16", font=("Times New Roman", 12)).place(x = 1690, y= 565)
    data39_ = Label(text = "17", font=("Times New Roman", 12)).place(x = 1690, y= 600)
    data39_ = Label(text = "18", font=("Times New Roman", 12)).place(x = 1690, y= 635)
    data39_ = Label(text = "19", font=("Times New Roman", 12)).place(x = 1690, y= 670)
    data39_ = Label(text = "20", font=("Times New Roman", 12)).place(x = 1690, y= 705)
    data39_ = Label(text = "21", font=("Times New Roman", 12)).place(x = 1690, y= 740)
    data39_ = Label(text = "22", font=("Times New Roman", 12)).place(x = 1690, y= 775)
    data39_ = Label(text = "23", font=("Times New Roman", 12)).place(x = 1690, y= 810)
    data39_ = Label(text = "24", font=("Times New Roman", 12)).place(x = 1690, y= 845)
    data39_ = Label(text = "25", font=("Times New Roman", 12)).place(x = 1690, y= 880)
    '''
    helv35=font.Font(family='Times New Roman', size=15)


    #text = Text(width = 65, height = 5)
    text =ScrolledText(width = 85, height = 5)
    #text.place(x =700, y = 470)

    # Create a text entry box 
    # for filling or typing the information.         

    lat1 = Entry(root)
    longt1 = Entry(root)
    alt1 = Entry(root)   
    bat1 = Entry(root)   
    arm1=Entry(root)
    mode1=Entry(root)
    airs1 = Entry(root)
    """
    T_lat1 = Entry(root)
    T_lon1 = Entry(root)
    T_score1 = Entry(root)
    """

    lat1.config(font=helv35)
    longt1.config(font=helv35)
    alt1.config(font=helv35)
    bat1.config(font=helv35)
    mode1.config(font=helv35)
    arm1.config(font=helv35)
    airs1.config(font=helv35)




    lat2 = Entry(root)
    longt2 = Entry(root)
    alt2 = Entry(root)
    bat2 = Entry(root)   
    mode2=Entry(root)
    airs2 = Entry(root)
    arm2=Entry(root)
    arm2.config(font=helv35)
    lat2.config(font=helv35)
    longt2.config(font=helv35)
    alt2.config(font=helv35)
    bat2.config(font=helv35)
    mode2.config(font=helv35)
    airs2.config(font=helv35)


    lat3 = Entry(root)
    longt3 = Entry(root)
    alt3 = Entry(root)
    bat3 = Entry(root)   
    mode3=Entry(root)
    arm3=Entry(root)
    airs3 = Entry(root)
    arm3.config(font=helv35)
    lat3.config(font=helv35)
    longt3.config(font=helv35)
    alt3.config(font=helv35)
    airs3.config(font=helv35)
    bat3.config(font=helv35)
   
    mode3.config(font=helv35)
   

    lat4 = Entry(root)
    longt4 = Entry(root)
    alt4 = Entry(root)

    bat4 = Entry(root)   
    arm4=Entry(root)
    mode4=Entry(root)
    arm4.config(font=helv35)
 
 
    lat4.config(font=helv35)
    longt4.config(font=helv35)
    alt4.config(font=helv35)
    airs4 = Entry(root)
    bat4.config(font=helv35)
    airs4.config(font=helv35)
    mode4.config(font=helv35)
 

    lat5 = Entry(root)
    longt5 = Entry(root)
    alt5 = Entry(root)
    arm5=Entry(root)
    bat5 = Entry(root)   

    mode5=Entry(root)
   
    lat5.config(font=helv35)
    longt5.config(font=helv35)
    alt5.config(font=helv35)
    airs5 = Entry(root)
    bat5.config(font=helv35)
    arm5.config(font=helv35)
    airs5.config(font=helv35)
    mode5.config(font=helv35)
 

    lat6 = Entry(root)
    longt6 = Entry(root)
    alt6 = Entry(root)
    bat6 = Entry(root)   
    airs6 = Entry(root)
    mode6=Entry(root)
    arm6=Entry(root)
    lat6.config(font=helv35)
    longt6.config(font=helv35)
    arm6.config(font=helv35)
    bat6.config(font=helv35)
    airs6.config(font=helv35)
    mode6.config(font=helv35)
    

    lat7 = Entry(root)
    longt7 = Entry(root)
    alt7 = Entry(root)
    airs7 = Entry(root)
    bat7 = Entry(root)   
    arm7=Entry(root)
    mode7=Entry(root)
 
    lat7.config(font=helv35)
    longt7.config(font=helv35)
    alt7.config(font=helv35)
    arm7.config(font=helv35)
    bat7.config(font=helv35)
    airs7.config(font=helv35)
    mode7.config(font=helv35)
  

    lat8 = Entry(root)
    longt8 = Entry(root)
    alt8 = Entry(root)
    arm8=Entry(root)
    bat8 = Entry(root)   
    airs8 = Entry(root)
    mode8=Entry(root)
    arm8.config(font=helv35)
    lat8.config(font=helv35)
    longt8.config(font=helv35)
    alt8.config(font=helv35)
    bat8.config(font=helv35)
    airs8.config(font=helv35)
    mode8.config(font=helv35)
    
    lat9 = Entry(root)
    longt9 = Entry(root)
    alt9 = Entry(root)
    arm9=Entry(root)
    bat9 = Entry(root)   
    mode9=Entry(root)
    airs9 = Entry(root)
    lat9.config(font=helv35)
    longt9.config(font=helv35)
    alt9.config(font=helv35)
    arm9.config(font=helv35)
    bat9.config(font=helv35)
    airs9.config(font=helv35)
    mode9.config(font=helv35)
    
    lat10 = Entry(root)
    longt10 = Entry(root)
    alt10 = Entry(root)
    arm10=Entry(root)
    bat10 = Entry(root)   
    
    mode10=Entry(root)
    airs10 = Entry(root)
    arm10.config(font=helv35)
    lat10.config(font=helv35)
    longt10.config(font=helv35)
    alt10.config(font=helv35)
    airs10.config(font=helv35)
    bat10.config(font=helv35)
    
    mode10.config(font=helv35)
    
    lat11 = Entry(root)
    longt11 = Entry(root)
    alt11 = Entry(root)
    
    bat11 = Entry(root)   
    arm11=Entry(root)
    mode11=Entry(root)
    airs11 = Entry(root)
    airs11.config(font=helv35)
    lat11.config(font=helv35)
    longt11.config(font=helv35)
    alt11.config(font=helv35)
    arm11.config(font=helv35)
    bat11.config(font=helv35)
    
    mode11.config(font=helv35)
    

    lat12 = Entry(root)
    longt12 = Entry(root)
    alt12 = Entry(root)
    arm12=Entry(root)
    bat12 = Entry(root)   
    airs12 = Entry(root)
    mode12=Entry(root)
    airs12.config(font=helv35)
    arm12.config(font=helv35)
    lat12.config(font=helv35)
    longt12.config(font=helv35)
    alt12.config(font=helv35)
    
    mode12.config(font=helv35)
    

    lat13 = Entry(root)
    longt13 = Entry(root)
    alt13 = Entry(root)
    arm13=Entry(root)
    bat13 = Entry(root)   
    
    mode13=Entry(root)
    arm13.config(font=helv35)
    lat13.config(font=helv35)
    longt13.config(font=helv35)
    
    alt13.config(font=helv35)
    airs13 = Entry(root)
    airs13.config(font=helv35)
    bat13.config(font=helv35)
    
    mode13.config(font=helv35)
    
    lat14 = Entry(root)
    longt14 = Entry(root)
    alt14 = Entry(root)
    arm14=Entry(root)
    bat14 = Entry(root)   
    airs14 = Entry(root)
    mode14=Entry(root)
   
    lat14.config(font=helv35)
    longt14.config(font=helv35)
    alt14.config(font=helv35)
    arm14.config(font=helv35)
    airs14.config(font=helv35)
    bat14.config(font=helv35)   
    mode14.config(font=helv35)
   
    lat15 = Entry(root)
    longt15 = Entry(root)
    alt15 = Entry(root)
    airs15 = Entry(root)
    bat15 = Entry(root)   
    arm15=Entry(root)
    mode15=Entry(root)
    arm15.config(font=helv35)
    lat15.config(font=helv35)
    longt15.config(font=helv35)
    airs15.config(font=helv35)
    alt15.config(font=helv35)
    bat15.config(font=helv35)
    mode15.config(font=helv35)    
    '''
    lat16 = Entry(root)
    longt16 = Entry(root)
    alt16 = Entry(root)
    airs16 = Entry(root)   
    sat16 = Entry(root)   
    bat16 = Entry(root)   
    #link16 = Entry(root)
    #rssi16=Entry(root)
    ##mag16=Entry(root)
    ekf16=Entry(root)
    arm16=Entry(root)
    mode16=Entry(root)
    s_lat16=Entry()
    s_lon16=Entry()
    g_lat16=Entry()
    g_lon16=Entry()
    g_alt16=Entry()
    """
    T_lat16 = Entry(root)
    T_lon16 = Entry(root)
    T_score16 = Entry(root)
    """
    lat16.config(font=helv35)
    longt16.config(font=helv35)
    alt16.config(font=helv35)
    airs16.config(font=helv35)
    sat16.config(font=helv35)
    bat16.config(font=helv35)
    #link16.config(font=helv35)
    #rssi16.config(font=helv35)
    ##mag16.config(font=helv35)
    ekf16.config(font=helv35)
    arm16.config(font=helv35)
    mode16.config(font=helv35)
    s_lat16.config(font=helv35)
    s_lon16.config(font=helv35)
    g_lat16.config(font=helv35)
    g_lon16.config(font=helv35)
    g_alt16.config(font=helv35)
    """
    T_lat16.config(font=helv35)
    T_lon16.config(font=helv35)
    T_score16.config(font=helv35)
    """

    lat17 = Entry(root)
    longt17 = Entry(root)
    alt17 = Entry(root)
    airs17 = Entry(root)   
    sat17 = Entry(root)   
    bat17 = Entry(root)   
    #link17 = Entry(root)
    #rssi17=Entry(root)
    ##mag17=Entry(root)
    ekf17=Entry(root)
    arm17=Entry(root)
    mode17=Entry(root)
    s_lat17=Entry()
    s_lon17=Entry()
    g_lat17=Entry()
    g_lon17=Entry()
    g_alt17=Entry()
    """
    T_lat17 = Entry(root)
    T_lon17 = Entry(root)
    T_score17 = Entry(root)
    """
    lat17.config(font=helv35)
    longt17.config(font=helv35)
    alt17.config(font=helv35)
    airs17.config(font=helv35)
    sat17.config(font=helv35)
    bat17.config(font=helv35)
    #link17.config(font=helv35)
    #rssi17.config(font=helv35)
    ##mag17.config(font=helv35)
    ekf17.config(font=helv35)
    arm17.config(font=helv35)
    mode17.config(font=helv35)
    s_lat17.config(font=helv35)
    s_lon17.config(font=helv35)
    g_lat17.config(font=helv35)
    g_lon17.config(font=helv35)
    g_alt17.config(font=helv35)
    """
    T_lat17.config(font=helv35)
    T_lon17.config(font=helv35)
    T_score17.config(font=helv35)
    """


    lat18 = Entry(root)
    longt18 = Entry(root)
    alt18 = Entry(root)
    airs18 = Entry(root)   
    sat18 = Entry(root)   
    bat18 = Entry(root)   
    #link18 = Entry(root)
    #rssi18=Entry(root)
    ##mag18=Entry(root)
    ekf18=Entry(root)
    arm18=Entry(root)
    mode18=Entry(root)
    s_lat18=Entry()
    s_lon18=Entry()
    g_lat18=Entry()
    g_lon18=Entry()
    g_alt18=Entry()
    """
    T_lat18 = Entry(root)
    T_lon18 = Entry(root)
    T_score18 = Entry(root)
    """
    lat18.config(font=helv35)
    longt18.config(font=helv35)
    alt18.config(font=helv35)
    airs18.config(font=helv35)
    sat18.config(font=helv35)
    bat18.config(font=helv35)
    #link18.config(font=helv35)
    #rssi18.config(font=helv35)
    ##mag18.config(font=helv35)
    ekf18.config(font=helv35)
    arm18.config(font=helv35)
    mode18.config(font=helv35)
    s_lat18.config(font=helv35)
    s_lon18.config(font=helv35)
    g_lat18.config(font=helv35)
    g_lon18.config(font=helv35)
    g_alt18.config(font=helv35)
    """
    T_lat18.config(font=helv35)
    T_lon18.config(font=helv35)
    T_score18.config(font=helv35)
    """

    lat19 = Entry(root)
    longt19 = Entry(root)
    alt19 = Entry(root)
    airs19 = Entry(root)   
    sat19 = Entry(root)   
    bat19 = Entry(root)   
    #link19 = Entry(root)
    #rssi19=Entry(root)
    ##mag19=Entry(root)
    ekf19=Entry(root)
    arm19=Entry(root)
    mode19=Entry(root)
    s_lat19=Entry()
    s_lon19=Entry()
    g_lat19=Entry()
    g_lon19=Entry()
    g_alt19=Entry()
    """
    T_lat19 = Entry(root)
    T_lon19 = Entry(root)
    T_score19 = Entry(root)
    """
    lat19.config(font=helv35)
    longt19.config(font=helv35)
    alt19.config(font=helv35)
    airs19.config(font=helv35)
    sat19.config(font=helv35)
    bat19.config(font=helv35)
    #link19.config(font=helv35)
    #rssi19.config(font=helv35)
    ##mag19.config(font=helv35)
    ekf19.config(font=helv35)
    arm19.config(font=helv35)
    mode19.config(font=helv35)
    s_lat19.config(font=helv35)
    s_lon19.config(font=helv35)
    g_lat19.config(font=helv35)
    g_lon19.config(font=helv35)
    g_alt19.config(font=helv35)
    """
    T_lat19.config(font=helv35)
    T_lon19.config(font=helv35)
    T_score19.config(font=helv35)
    """

    lat20 = Entry(root)
    longt20 = Entry(root)
    alt20 = Entry(root)
    airs20 = Entry(root)   
    sat20 = Entry(root)   
    bat20 = Entry(root)   
    #link20 = Entry(root)
    #rssi20=Entry(root)
    ##mag20=Entry(root)
    ekf20=Entry(root)
    arm20=Entry(root)
    mode20=Entry(root)
    s_lat20=Entry()
    s_lon20=Entry()
    g_lat20=Entry()
    g_lon20=Entry()
    g_alt20=Entry()
    """
    T_lat20 = Entry(root)
    T_lon20 = Entry(root)
    T_score20 = Entry(root)
    """
    lat20.config(font=helv35)
    longt20.config(font=helv35)
    alt20.config(font=helv35)
    airs20.config(font=helv35)
    sat20.config(font=helv35)
    bat20.config(font=helv35)
    #link20.config(font=helv35)
    #rssi20.config(font=helv35)
    ##mag20.config(font=helv35)
    ekf20.config(font=helv35)
    arm20.config(font=helv35)
    mode20.config(font=helv35)
    s_lat20.config(font=helv35)
    s_lon20.config(font=helv35)
    g_lat20.config(font=helv35)
    g_lon20.config(font=helv35)
    g_alt20.config(font=helv35)
    """
    T_lat20.config(font=helv35)
    T_lon20.config(font=helv35)
    T_score20.config(font=helv35)
    """
    lat21 = Entry(root)
    longt21 = Entry(root)
    alt21 = Entry(root)
    airs21 = Entry(root)   
    sat21 = Entry(root)   
    bat21 = Entry(root)   
    #link21 = Entry(root)
    #rssi21=Entry(root)
    ##mag21=Entry(root)
    ekf21=Entry(root)
    arm21=Entry(root)
    mode21=Entry(root)
    s_lat21=Entry()
    s_lon21=Entry()
    g_lat21=Entry()
    g_lon21=Entry()
    g_alt21=Entry()
    """
    T_lat21 = Entry(root)
    T_lon21 = Entry(root)
    T_score21 = Entry(root)
    """
    lat21.config(font=helv35)
    longt21.config(font=helv35)
    alt21.config(font=helv35)
    airs21.config(font=helv35)
    sat21.config(font=helv35)
    bat21.config(font=helv35)
    #link21.config(font=helv35)
    #rssi21.config(font=helv35)
    ##mag21.config(font=helv35)
    ekf21.config(font=helv35)
    arm21.config(font=helv35)
    mode21.config(font=helv35)
    s_lat21.config(font=helv35)
    s_lon21.config(font=helv35)
    g_lat21.config(font=helv35)
    g_lon21.config(font=helv35)
    g_alt21.config(font=helv35)
    """
    T_lat21.config(font=helv35)
    T_lon21.config(font=helv35)
    T_score21.config(font=helv35)
    """

    lat22 = Entry(root)
    longt22 = Entry(root)
    alt22 = Entry(root)
    airs22 = Entry(root)   
    sat22 = Entry(root)   
    bat22 = Entry(root)   
    #link22 = Entry(root)
    #rssi22=Entry(root)
    ##mag22=Entry(root)
    ekf22=Entry(root)
    arm22=Entry(root)
    mode22=Entry(root)
    s_lat22=Entry()
    s_lon22=Entry()
    g_lat22=Entry()
    g_lon22=Entry()
    g_alt22=Entry()
    """
    T_lat22 = Entry(root)
    T_lon22 = Entry(root)
    T_score22 = Entry(root)
    """
    lat22.config(font=helv35)
    longt22.config(font=helv35)
    alt22.config(font=helv35)
    airs22.config(font=helv35)
    sat22.config(font=helv35)
    bat22.config(font=helv35)
    #link22.config(font=helv35)
    #rssi22.config(font=helv35)
    ##mag22.config(font=helv35)
    ekf22.config(font=helv35)
    arm22.config(font=helv35)
    mode22.config(font=helv35)
    s_lat22.config(font=helv35)
    s_lon22.config(font=helv35)
    g_lat22.config(font=helv35)
    g_lon22.config(font=helv35)
    g_alt22.config(font=helv35)
    """
    T_lat22.config(font=helv35)
    T_lon22.config(font=helv35)
    T_score22.config(font=helv35)
    """
    lat23 = Entry(root)
    longt23 = Entry(root)
    alt23 = Entry(root)
    airs23 = Entry(root)   
    sat23 = Entry(root)   
    bat23 = Entry(root)   
    #link23 = Entry(root)
    #rssi23=Entry(root)
    ##mag23=Entry(root)
    ekf23=Entry(root)
    arm23=Entry(root)
    mode23=Entry(root)
    s_lat23=Entry()
    s_lon23=Entry()
    g_lat23=Entry()
    g_lon23=Entry()
    g_alt23=Entry()
    """
    T_lat23 = Entry(root)
    T_lon23 = Entry(root)
    T_score23 = Entry(root)
    """
    lat23.config(font=helv35)
    longt23.config(font=helv35)
    alt23.config(font=helv35)
    airs23.config(font=helv35)
    sat23.config(font=helv35)
    bat23.config(font=helv35)
    #link23.config(font=helv35)
    #rssi23.config(font=helv35)
    ##mag23.config(font=helv35)
    ekf23.config(font=helv35)
    arm23.config(font=helv35)
    mode23.config(font=helv35)
    s_lat23.config(font=helv35)
    s_lon23.config(font=helv35)
    g_lat23.config(font=helv35)
    g_lon23.config(font=helv35)
    g_alt23.config(font=helv35)
    """
    T_lat23.config(font=helv35)
    T_lon23.config(font=helv35)
    T_score23.config(font=helv35)
    """

    lat24 = Entry(root)
    longt24 = Entry(root)
    alt24 = Entry(root)
    airs24 = Entry(root)   
    sat24 = Entry(root)   
    bat24 = Entry(root)   
    #link24 = Entry(root)
    #rssi24=Entry(root)
    ##mag24=Entry(root)
    ekf24=Entry(root)
    arm24=Entry(root)
    mode24=Entry(root)
    s_lat24=Entry()
    s_lon24=Entry()
    g_lat24=Entry()
    g_lon24=Entry()
    g_alt24=Entry()
    """
    T_lat24 = Entry(root)
    T_lon24 = Entry(root)
    T_score24 = Entry(root)
    """
    lat24.config(font=helv35)
    longt24.config(font=helv35)
    alt24.config(font=helv35)
    airs24.config(font=helv35)
    sat24.config(font=helv35)
    bat24.config(font=helv35)
    #link24.config(font=helv35)
    #rssi24.config(font=helv35)
    ##mag24.config(font=helv35)
    ekf24.config(font=helv35)
    arm24.config(font=helv35)
    mode24.config(font=helv35)
    s_lat24.config(font=helv35)
    s_lon24.config(font=helv35)
    g_lat24.config(font=helv35)
    g_lon24.config(font=helv35)
    g_alt24.config(font=helv35)
    """
    T_lat24.config(font=helv35)
    T_lon24.config(font=helv35)
    T_score24.config(font=helv35)
    """

    lat25 = Entry(root)
    longt25 = Entry(root)
    alt25 = Entry(root)
    airs25 = Entry(root)   
    sat25 = Entry(root)   
    bat25 = Entry(root)   
    #link25 = Entry(root)
    #rssi25=Entry(root)
    ##mag25=Entry(root)
    ekf25=Entry(root)
    arm25=Entry(root)
    mode25=Entry(root)
    s_lat25=Entry()
    s_lon25=Entry()
    g_lat25=Entry()
    g_lon25=Entry()
    g_alt25=Entry()
    """
    T_lat25 = Entry(root)
    T_lon25 = Entry(root)
    T_score25 = Entry(root)
    """
    lat25.config(font=helv35)
    longt25.config(font=helv35)
    alt25.config(font=helv35)
    airs25.config(font=helv35)
    sat25.config(font=helv35)
    bat25.config(font=helv35)
    #link25.config(font=helv35)
    #rssi25.config(font=helv35)
    ##mag25.config(font=helv35)
    ekf25.config(font=helv35)
    arm25.config(font=helv35)
    mode25.config(font=helv35)
    s_lat25.config(font=helv35)
    s_lon25.config(font=helv35)
    g_lat25.config(font=helv35)
    g_lon25.config(font=helv35)
    g_alt25.config(font=helv35)
    """
    T_lat25.config(font=helv35)
    T_lon25.config(font=helv35)
    T_score25.config(font=helv35)
    """
    
    THD = Entry(root)
    THD.config(font=helv35)
    THD.place(x=1180+70+100-900,y=1000+15+35+2,height=25,width = 40)
    '''

    ##vehicle_connect()
    #helv35=font.Font(family='Times New Roman', size=15)
    var1 = StringVar(root)
    var1.set("STABILIZE") # initial value
    option1 = OptionMenu(root,var1, "STABILIZE", "LOITER", "AUTO", "GUIDED", "RTL","LAND")
    option1.config(font=helv35)
    option1.place(x=713,y=40,height=30,width = 140)
	
    var2= StringVar(root)
    var2.set("STABILIZE") # initial value
    option2 = OptionMenu(root,var2, "STABILIZE", "LOITER", "AUTO", "GUIDED", "RTL","LAND")
    option2.config(font=helv35)
    option2.place(x=713,y=75,height=30,width = 140)

    var3 = StringVar(root)
    var3.set("STABILIZE") # initial value
    option3 = OptionMenu(root,var3, "STABILIZE", "LOITER", "AUTO", "GUIDED", "RTL","LAND")
    option3.config(font=helv35)
    option3.place(x=713,y=110,height=30,width = 140)

    var4 = StringVar(root)
    var4.set("STABILIZE") # initial value
    option4 = OptionMenu(root,var4, "STABILIZE", "LOITER", "AUTO", "GUIDED", "RTL","LAND")
    option4.config(font=helv35)
    option4.place(x=713,y=145,height=30,width = 140)

    var5 = StringVar(root)
    var5.set("STABILIZE") # initial value
    option5 = OptionMenu(root,var5, "STABILIZE", "LOITER", "AUTO", "GUIDED", "RTL","LAND")
    option5.config(font=helv35)
    option5.place(x=713,y=180,height=30,width = 140)

    var6= StringVar(root)
    var6.set("STABILIZE") # initial value
    option6 = OptionMenu(root,var6, "STABILIZE", "LOITER", "AUTO", "GUIDED", "RTL","LAND")
    option6.config(font=helv35)
    option6.place(x=713,y=215,height=30,width = 140)

    var7 = StringVar(root)
    var7.set("STABILIZE") # initial value
    option7 = OptionMenu(root,var7, "STABILIZE", "LOITER", "AUTO", "GUIDED", "RTL","LAND")
    option7.config(font=helv35)
    option7.place(x=713,y=250,height=30,width = 140)

    var8 = StringVar(root)
    var8.set("STABILIZE") # initial value
    option8 = OptionMenu(root,var8, "STABILIZE", "LOITER", "AUTO", "GUIDED", "RTL","LAND")
    option8.config(font=helv35)
    option8.place(x=713,y=285,height=30,width = 140)

    var9 = StringVar(root)
    var9.set("STABILIZE") # initial value
    option9 = OptionMenu(root,var9, "STABILIZE", "LOITER", "AUTO", "GUIDED", "RTL","LAND")
    option9.config(font=helv35)
    option9.place(x=713,y=320,height=30,width = 140)

    var10 = StringVar(root)
    var10.set("STABILIZE") # initial value
    option10 = OptionMenu(root,var10, "STABILIZE", "LOITER", "AUTO", "GUIDED", "RTL","LAND")
    option10.config(font=helv35)
    option10.place(x=713,y=355,height=30,width = 140)

    var11 = StringVar(root)
    var11.set("STABILIZE") # initial value
    option11 = OptionMenu(root,var11, "STABILIZE", "LOITER", "AUTO", "GUIDED", "RTL","LAND")
    option11.config(font=helv35)
    option11.place(x=713,y=390,height=30,width = 140)

    var12= StringVar(root)
    var12.set("STABILIZE") # initial value
    option12 = OptionMenu(root,var12, "STABILIZE", "LOITER", "AUTO", "GUIDED", "RTL","LAND")
    option12.config(font=helv35)
    option12.place(x=713,y=425,height=30,width = 140)

    var13 = StringVar(root)
    var13.set("STABILIZE") # initial value
    option13 = OptionMenu(root,var13, "STABILIZE", "LOITER", "AUTO", "GUIDED", "RTL","LAND")
    option13.config(font=helv35)
    option13.place(x=713,y=460,height=30,width = 140)

    var14 = StringVar(root)
    var14.set("STABILIZE") # initial value
    option14 = OptionMenu(root,var14, "STABILIZE", "LOITER", "AUTO", "GUIDED", "RTL","LAND")
    option14.config(font=helv35)
    option14.place(x=713,y=495,height=30,width = 140)

    var15 = StringVar(root)
    var15.set("STABILIZE") # initial value
    option15 = OptionMenu(root,var15, "STABILIZE", "LOITER", "AUTO", "GUIDED", "RTL","LAND")
    option15.config(font=helv35)
    option15.place(x=713,y=530,height=30,width = 140)
    '''

    var16= StringVar(root)
    var16.set("STABILIZE") # initial value
    option16 = OptionMenu(root,var16, "STABILIZE", "LOITER", "AUTO", "GUIDED", "RTL","LAND")
    option16.config(font=helv35)
    option16.place(x=550,y=565,height=30,width = 140)

    var17 = StringVar(root)
    var17.set("STABILIZE") # initial value
    option17 = OptionMenu(root,var17, "STABILIZE", "LOITER", "AUTO", "GUIDED", "RTL","LAND")
    option17.config(font=helv35)
    option17.place(x=550,y=600,height=30,width = 140)

    var18 = StringVar(root)
    var18.set("STABILIZE") # initial value
    option18 = OptionMenu(root,var18, "STABILIZE", "LOITER", "AUTO", "GUIDED", "RTL","LAND")
    option18.config(font=helv35)
    option18.place(x=550,y=635,height=30,width = 140)

    var19 = StringVar(root)
    var19.set("STABILIZE") # initial value
    option19 = OptionMenu(root,var19, "STABILIZE", "LOITER", "AUTO", "GUIDED", "RTL","LAND")
    option19.config(font=helv35)
    option19.place(x=550,y=670,height=30,width = 140)

    var20 = StringVar(root)
    var20.set("STABILIZE") # initial value
    option20 = OptionMenu(root,var20, "STABILIZE", "LOITER", "AUTO", "GUIDED", "RTL","LAND")
    option20.config(font=helv35)
    option20.place(x=550,y=705,height=30,width = 140)

    var21= StringVar(root)
    var21.set("STABILIZE") # initial value
    option21 = OptionMenu(root,var16, "STABILIZE", "LOITER", "AUTO", "GUIDED", "RTL","LAND")
    option21.config(font=helv35)
    option21.place(x=550,y=740,height=30,width = 140)

    var22 = StringVar(root)
    var22.set("STABILIZE") # initial value
    option22 = OptionMenu(root,var17, "STABILIZE", "LOITER", "AUTO", "GUIDED", "RTL","LAND")
    option22.config(font=helv35)
    option22.place(x=550,y=775,height=30,width = 140)

    var23 = StringVar(root)
    var23.set("STABILIZE") # initial value
    option23 = OptionMenu(root,var18, "STABILIZE", "LOITER", "AUTO", "GUIDED", "RTL","LAND")
    option23.config(font=helv35)
    option23.place(x=550,y=810,height=30,width = 140)

    var24 = StringVar(root)
    var24.set("STABILIZE") # initial value
    option24 = OptionMenu(root,var19,"STABILIZE", "LOITER", "AUTO", "GUIDED", "RTL","LAND")
    option24.config(font=helv35)
    option24.place(x=550,y=845,height=30,width = 140)

    var25 = StringVar(root)
    var25.set("STABILIZE") # initial value
    option25 = OptionMenu(root,var20, "STABILIZE", "LOITER", "AUTO", "GUIDED", "RTL","LAND")
    option25.config(font=helv35)
    option25.place(x=550,y=880,height=30,width = 140)
    '''

    vehicle_connect = Button(text = "Connect", command = vehicle_connect, font=("Times New Roman", 13)).place(x = 885, y = 950)
    disconnect = Button(text = "Disconnect", command = disconnect, font=("Times New Roman", 13)).place(x = 980, y = 950)

    Reconnect = Button(text = "Reconnect =", command = Reconnect, width = 8, font=("Times New Roman", 13)).place(x = 1095, y = 950)
    Reconnectset_entry = Entry()
    Reconnectset_entry.place(x = 1205, y = 955, width = 50, height = 30)

    '''
    varCheckButton1 = tkinter.IntVar()
    tkCheckButton = tkinter.Checkbutton(
    root,
    text="MANUAL",
    variable=varCheckButton1,
    command=setCheckButtonText1, font=("Times New Roman", 11)).place(x = 1260, y = 940)
    
   
    varCheckButton2 = tkinter.IntVar()
    tkCheckButton = tkinter.Checkbutton(
    root,
    text="AUTONUMUS",
    variable=varCheckButton2,
    command=setCheckButtonText2, font=("Times New Roman", 11)).place(x = 1260, y = 965)
    '''
    """
    data313_ = Label(text = "SEMI", font=("Times New Roman", 13)).place(x = 1260+6, y = 940)
    data313_ = Label(text = "AUTONUMUS", font=("Times New Roman", 13)).place(x = 1260+6, y = 965)

    varCheckButton3 = tkinter.IntVar()
    tkCheckButton = tkinter.Checkbutton(
    root,
    text="self_heal_alt",
    variable=varCheckButton3,
    command=setCheckButtonText3, font=("Times New Roman", 10)).place(x = 1000+70+50, y = 1000+15+2)
    """

    """        
    root = Tk()
    var = IntVar()

    R1 = Radiobutton(root, text="Option 1", variable=var, value=1, command=sel)
    R1.pack( anchor = W )
    """

    """
    altitude = Button(text = "ALT", command = altitude, width = 3, font=("Times New Roman", 15)).place(x = 20, y = 680)
    aggr = Button(text = "V-formation", command = aggr, width = 7, font=("Times New Roman", 15)).place(x = 80, y =680)
    ##disp = Button(text = "Dispersion", command = disp, width = 7, font=("Times New Roman", 15)).place(x = 270, y = 680)
    data311_ = Label(text = "xoffset", font=("Times New Roman", 15)).place(x = 170, y= 660)
    xoffset_entry = Entry()
    xoffset_entry.place(x = 170, y = 680, width = 40, height = 30)
    data312_ = Label(text = "yoffset", font=("Times New Roman", 15)).place(x = 230, y= 660)
    yoffset_entry = Entry()
    yoffset_entry.place(x = 230, y = 680, width = 40, height = 30)
    pause = Button(text = "Pause", command = pause, width = 5, font=("Times New Roman", 15)).place(x = 350, y = 680)
    waypoint = Button(text = "Wp_upload", command = waypoint, width = 7, font=("Times New Roman", 15)).place(x = 450, y = 680)
    
    data313_ = Button(text = "no.of.uavs", font=("Times New Roman", 15)).place(x=1400, y=900+120)
    no_of_uavs_entry = Entry()
    no_of_uavs_entry.place(x = 1400+100,  y= 900+120,width = 50)
    """

    """
    arm_all = Button(text = "Arm_all", command = arm_all, width = 5, font=("Times New Roman", 15)).place(x = 710, y = 20)
    takeoff_all = Button(text = "Takeoff_all", command = takeoff_all, width = 6, font=("Times New Roman", 15)).place(x = 790, y = 20)
    data_entry = Entry()
    data_entry.place(x = 875, y = 20,width = 50, height = 25)
    auto = Button(text = "Auto_all", command = auto, width = 5, font=("Times New Roman", 15)).place(x =935, y = 20)
    rtl = Button(text = "RTL_all", command = rtl , width = 5, font=("Times New Roman", 15)).place(x = 1010, y = 20)
    """


    Button(text = "Takeoff", command = takeoff_socket, width = 5, font=("Times New Roman", 13)).place(x = 10, y = 950)

    takeoff_entry = Entry()
    takeoff_entry.place(x = 90, y = 955, width = 30, height = 30)

    Remove = Button(text = "Remove:", command = Remove, width = 6, font=("Times New Roman", 13)).place(x = 90+35, y = 950)
    Removeset_entry = Entry()
    Removeset_entry.place(x = 210, y = 955, width = 50, height = 30)


    ##search_new = Button(text = "search", command = search_new, width = 5, font=("Times New Roman", 15)).place(x = 15, y = 950)
    ##takeoff_all = Button(text = "Takeoff_2", command = takeoff_all, width = 8, font=("Times New Roman", 15)).place(x = 110, y = 950)
    Button(text = "move", command = search, width = 3, font=("Times New Roman", 13)).place(x =270+5, y = 950)
    #data_entry = Entry()
    #data_entry.place(x = 290, y = 950,width = 50, height = 30)
    Button(text = "Auto", command = auto, width = 3, font=("Times New Roman", 13)).place(x =330+5, y = 950)
    Button(text = "RTL", command = rtl_socket , width = 3, font=("Times New Roman", 13)).place(x = 390+5, y = 950)

    Button(text = "LAND", command = land_socket, width = 5, font=("Times New Roman", 13)).place(x = 450+5, y = 950)

    Button(text = "Start", command = start_socket, width = 5, font=("Times New Roman", 13)).place(x = 490+40, y = 950)
    Button(text = "Return", command = return_socket, width =6, font=("Times New Roman", 13)).place(x = 860+70+15+10, y = 985)
    Button(text = "Clear_CSV", command = clear_csv, width =7, font=("Times New Roman", 13)).place(x = 460+150, y = 1025)
   # Button(text = "Resume", command = resume_socket, width = 6, font=("Times New Roman", 13)).place(x = 860+70+15+10, y = 1020)
    Button(text = "Share_data", command = share_data_func, width = 8, font=("Times New Roman", 13)).place(x = 1120+70+50+15+10, y = 950)
    Button(text = "Start1", command = start1_socket, width = 6, font=("Times New Roman", 13)).place(x = 610, y = 950)
    Button(text = "Disperse", command = disperse_socket, width =5, font=("Times New Roman", 13)).place(x = 700, y = 950)
    Button(text = "Search", command = search_socket, width = 5, font=("Times New Roman", 13)).place(x = 795, y = 950)
    Button(text = "Home", command = home_socket, width = 7, font=("Times New Roman", 13)).place(x = 460+150, y =1000-15)
    Button(text = "Home_Goto", command = home_goto_socket, width = 8, font=("Times New Roman", 13)).place(x = 752+70, y = 1000-10-5)
     
    Button(text = "Aggregate", command = aggregate_socket, width = 7, font=("Times New Roman", 13)).place(x = 250+70+50, y =1000)
    Button(text = "Stop", command = stop_socket, width = 6, font=("Times New Roman", 13)).place(x = 960+70+30, y = 985)
    uavs= Label(text = "Model_no", font=("Times New Roman", 13)).place(x = 15, y= 730)
    selected_drone = tk.StringVar()
    dropdown = ttk.Combobox(root, textvariable=selected_drone)
    dropdown['values'] = follower_host_tuple
    print("follower_host_tuple",follower_host_tuple)
    dropdown.place(x=100, y=730,width=70)
    dropdown.current() 
    c_alt= Label(text = "Altitude", font=("Times New Roman", 13)).place(x = 300, y= 730)
    c_alt_entry= Entry()
    c_alt_entry.place(x = 370, y= 730,width = 40)
    target_ll= Label(text = "Target_value", font=("Times New Roman", 13)).place(x = 15, y= 800)
    target_entry= Entry()
    target_entry.place(x = 130, y= 800,width = 130)
    
    rtl_height= Label(text = "RTL_Height", font=("Times New Roman", 13)).place(x = 300, y= 800)
    rtl_height_entry= Entry()
    rtl_height_entry.place(x = 400, y= 800,width = 40)
    Button(text = "STRIKE", command = strike_target, width = 8, font=("Times New Roman", 13),fg='red').place(x = 120, y = 850)
    Button(text = "CANCEL MISSION", command = strike_stop, width = 14, font=("Times New Roman", 13),fg='red').place(x = 280, y = 850)
    '''
    uavs= Label(text = "UAVS =", font=("Times New Roman", 13)).place(x = 15, y= 750)
    uavs_entry= Entry()
    uavs_entry.place(x = 80,  y= 750,width = 50)
    '''
    area_covered_var = tk.StringVar()
    data="0.0"
    area_covered_var.set("Area Covered = " + data)  # Initialize with default value

# Create a Label widget to display the area covered
    area_covered_label = tk.Label(root, textvariable=area_covered_var, font=("Times New Roman", 13)).place(x = 752+650, y = 1025)
    #area_covered= Label(text = "Area Covered =", font=("Times New Roman", 13)).place(x = 15, y= 850)
    #area= Label(text = "UAVS =", font=("Times New Roman", 13)).place(x = 15, y= 750)
    #area_entry= Entry()
    #area_entry.place(x = 80,  y= 750,width = 50)
    '''
    Button(text = "Compute ", command = fetch_and_compute, width = 8, font=("Times New Roman", 13)).place(x = 180, y = 745)
    computed_entry= Entry()
    computed_entry.place(x = 290,  y= 750,width = 50)
    '''
    Button(text = "ARM", command = arm_all, width = 3, font=("Times New Roman", 13)).place(x = 10, y = 1000)
    
    #Button(text = "Same ALt", command = same_alt_socket, width = 7, font=("Times New Roman", 13)).place(x = 1500, y = 940)

    Button(text = "Disarm", command = disarm_all, width = 5, font=("Times New Roman", 13)).place(x = 68, y = 1000)
    Button(text = "Specific_Bot_Goal", command = specific_bot_goal_socket, width = 15, font=("Times New Roman", 13)).place(x = 1600, y= 950)
    Button(text = "Move_bot_stop", command = move_bot_stop, width = 10, font=("Times New Roman", 13)).place(x = 1770, y= 950)
    Button(text = "Goal", command = goal_socket, width = 5, font=("Times New Roman", 13)).place(x = 1500, y= 950)
    Button(text = "Move Bot", command = move_bot_socket, width = 7, font=("Times New Roman", 13)).place(x = 1400, y= 950)
    goal_entry= Entry()
    goal_entry.place(x = 1500 , y= 990,width = 90)
    move_bot_entry= Entry()
    move_bot_entry.place(x = 1400,  y= 990,width = 90)
    
    specific_bot_goal_entry= Entry()
    specific_bot_goal_entry.place(x = 1600, y= 990)
    #Button(text = "Enter", command = enter, width = 8, font=("Times New Roman", 13)).place(x = 1730+20+35, y= 890+120)
    terminal_msg_label= Label(text = "Message", font=("Times New Roman", 13),fg='blue').place(x = 490, y= 670)
    #terminal_message_entry=Entry()
    terminal_message_entry=Text(root)
    terminal_message_entry.place(x = 490, y= 700,width=450,height=200)

    Button(text = "ALT_D", command = different_alt_socket , width = 4, font=("Times New Roman", 10)).place(x = 150, y = 985)
    Button(text = "ALT_S", command = same_alt_socket, width = 4, font=("Times New Roman", 10)).place(x = 150,y = 1015)
    
    alts_entry = Entry()
    alts_entry.place(x = 215, y = 1015, width = 30, height = 30)

    altd_entry = Entry()
    altd_entry.place(x = 215, y = 985, width = 30, height = 30)

    data111_ = Label(text = "Aoffset :", font=("Times New Roman", 13)).place(x = 250, y= 985)
    aoffset_entry = Entry()
    aoffset_entry.place(x = 310, y = 985, width = 30, height = 30)

    data112_ = Label(text = "S_alt :", font=("Times New Roman", 13)).place(x = 250, y= 1015)
    salt_entry = Entry()
    salt_entry.place(x = 310, y = 1015, width = 30, height = 30)
    
    data17_ = Label(text = "AIRPORTS", font=("Times New Roman", 15), fg = "red").place(x = 15, y= 600)
    
    def select_location():
    	    global fileloc,filename
	    selected_location = location_var.get()
	    print("selected_location",selected_location)
	    radio_var_val=radio_var.get()
	    print("radio_var_val",radio_var_val)
	    filename=selected_location+"_"+radio_var_val
	    print("filename",filename)
	    #update_plot()
	    fileloc="/home/dhaksha/Documents/strike_socket/dce_swarm_nav/swarm_tasks-main/swarm_tasks/Examples/basic_tasks/"
	    print("fileloc",fileloc)
	    file_name = fileloc+filename+".yaml"
	    print("filename!!!!!!!!!!!",file_name)
	    open_plot()
	    
    # Location Options
    location_options = {
	    "Tambaram", "dce","Sholavaram","Lucknow","Jammu"
	}
    airport_options=["airport", "runway", "taxiway", "apron"]

    # Select Location Label and Dropdown
    select_label = tk.Label(root, text="Select Location:")
    select_label.place(x=180, y=600)

    location_var = tk.StringVar()
    location_dropdown = ttk.Combobox(root, textvariable=location_var, values=list(location_options))
    location_dropdown.place(x=130, y=600)

    radio_var = tk.StringVar()
    
    #location_label.place(x = 150, y= 630)
    for i, option in enumerate(airport_options):
    	h=90*i
    	tk.Radiobutton(root, text=option, variable=radio_var, value=option).place(x = 330+h, y= 600)  	
	
    select_button = tk.Button(root, text="Select", command=select_location)
    select_button.place(x=700, y=600)

    # Location Label and Radio Buttons (Initially Hidden)
    location_label = tk.Label(root, text="Select Option:")
    
    
    
    '''

    selected_location = location_var.get()
    print("selected_location", selected_location)
    radio_var_val = radio_var.get()
    print("radio_var_val", radio_var_val)
    filename = selected_location + "_" + radio_var_val
    print("filename", filename)
    #update_plot()
    fileloc = "/home/dhaksha/Documents/socket/dce_swarm_nav/swarm_tasks-main/swarm_tasks/Examples/basic_tasks/"
    print("fileloc", fileloc)
    file_name = fileloc + filename + ".yaml"
    print("filename!!!!!!!!!!!", file_name)
    open_plot(file_name)

    # Add functionalities to be displayed conditionally
    # Only if a location and radio option are selected
    global uavs_entry
    uavs_label = tk.Label(root, text="UAVS =", font=("Times New Roman", 13))
    uavs_label.place(x=15, y=750)

    uavs_entry = tk.Entry(root)
    uavs_entry.place(x=80, y=750, width=50)

    compute_button = tk.Button(root, text="Compute", command=fetch_and_compute, width=8, font=("Times New Roman", 13))
    compute_button.place(x=180, y=745)

    global computed_entry
    computed_entry = tk.Entry(root)
    computed_entry.place(x=290, y=750, width=50)

    

    # Location Options
    location_options = {
    "G Square", "dce", "Sholavaram", "Lucknow", "Jammu"
    }
    airport_options = ["airport", "runway", "taxiway", "apron"]

    # Select Location Label and Dropdown
    select_label = tk.Label(root, text="Select Location:")
    select_label.place(x=180, y=600)

    location_var = tk.StringVar()
    location_dropdown = ttk.Combobox(root, textvariable=location_var, values=list(location_options))
    location_dropdown.place(x=130, y=600)

    radio_var = tk.StringVar()

	# Location Label and Radio Buttons (Initially Hidden)
    for i, option in enumerate(airport_options):
		h = 90 * i
		tk.Radiobutton(root, text=option, variable=radio_var, value=option).place(x=330 + h, y=600)

    select_button = tk.Button(root, text="Select", command=select_location)
    select_button.place(x=700, y=600)




   
    checkboxvalue1 = IntVar()
    checkboxvalue2 = IntVar()
    checkboxvalue3 = IntVar()
    checkboxvalue4 = IntVar()

    Checkbutton(root, variable=checkboxvalue1).place(x = 260+80, y = 982, width=20, height=20)
    Checkbutton(root, variable=checkboxvalue2).place(x = 260+80, y = 997, width=20, height=20)
    Checkbutton(root, variable=checkboxvalue3).place(x = 260+80, y = 1012, width=20, height=20)
    Checkbutton(root, variable=checkboxvalue4).place(x = 260+80, y = 1027, width=20, height=20)
    
    data511_ = Label(text = "T", font=("Times New Roman", 11)).place(x = 290+80, y= 982)
    data611_ = Label(text = "L", font=("Times New Roman", 11)).place(x = 290+80, y= 997)
    data711_ = Label(text = "S", font=("Times New Roman", 11)).place(x = 290+80, y= 1012)
    data811_ = Label(text = "C", font=("Times New Roman", 11)).place(x = 290+80, y= 1027)
    '''
    data311_ = Label(text = "xoffset =", font=("Times New Roman", 13)).place(x = 320+35+70+60, y= 990)
    xoffset_entry = Entry()
    xoffset_entry.place(x = 420+70+70, y = 985, width = 30, height = 30)
    data312_ = Label(text = "c_radius =", font=("Times New Roman", 13)).place(x = 320+35+70+60, y= 1010)
    cradius_entry = Entry()
    cradius_entry.place(x = 420+70+70, y = 1015, width = 30, height = 30)

    Button(text = "GUIDED", command = guided_all, width = 6, font=("Times New Roman", 13)).place(x = 460+70+80+110, y =1000-15)
    Button(text = "Home_Lock", command = home_lock, width = 7, font=("Times New Roman", 13)).place(x = 752+310, y = 1030,width=100,height=30)

    Button(text = "PLOT", command = update_plot, width = 7, font=("Times New Roman", 13)).place(x = 1000+70+50+15+10+20, y = 1000-10-2)
    Button(text = "Add link =", command = mavlink_add, width = 7, font=("Times New Roman", 13)).place(x = 460+70+80+110, y = 1025)
    addlink_entry = Entry()
    addlink_entry.place(x =  752+70, y = 1030,width=45)
    Button(text = "Remove link =", command = mavlink_remove, width = 10, font=("Times New Roman", 13)).place(x =752+70+60, y = 1025)
    removelink_entry = Entry()
    removelink_entry.place(x = 752+70+60+120, y = 1030,width=45)
    
    # Create a checkbox
    home_checkbox_var = tk.IntVar()
    home_checkbox = tk.Checkbutton(root, text="Home Point", variable=home_checkbox_var, command=toggle_home_point)
    home_checkbox.place(x = 752+500, y = 1030)
    
    goal_checkbox_var = tk.IntVar()
    goal_checkbox = tk.Checkbutton(root, text="Goal Point", variable=goal_checkbox_var, command=goal_points_func)
    goal_checkbox.place(x = 752+410, y = 1030)
    
    Button(text = "Remove Bot =", command = bot_remove, width = 10, font=("Times New Roman", 13)).place(x =1750, y = 1025)
    removebot_entry = Entry()
    removebot_entry.place(x = 1880, y = 1030,width=25)
    
   
   
   #home_pos_checkbox = tk.Checkbutton(root,text="Home Position",variable=show_home_pos,command= plot_home_pos(home_pos) )
    #home_pos_checkbox = tk.Checkbutton(root,text="Home Position",variable=show_home_pos,command=lambda: plot_home_pos(home_pos) if show_home_pos.get() else  update_plot())
    #home_pos_checkbox.place(x = 1000+70+50+15+10+20, y = 1025)	
    ##line = Button(text = "Line", command = line, width = 5, font=("Times New Roman", 15)).place(x = 240, y = 1000)

    '''
    ##data911_ = Label(text = "circle_pos :", font=("Times New Roman", 13)).place(x = 460, y= 1000)
    Button(text = "circle_pos", command = circle, width = 7, font=("Times New Roman", 10)).place(x = 460+70+80+15-5, y =1000-15)
    Button(text = "ROI_SET", command = ROI_OFFSET, width = 6, font=("Times New Roman", 9)).place(x = 752+70-30, y = 1000-10-5)

    Button(text = "ROI_HD", command = ROI_heading, width = 5, font=("Times New Roman", 9)).place(x = 752+70+25+10+20-15, y = 1000-10-5)
    roi_entry = Entry()
    roi_entry.place(x = 752+70+25+10+20-10+15+20+10+5, y = 1000-10-5, width = 30, height = 28)

    #Button(text = "Wp_upload", command = waypoint, width = 8, font=("Times New Roman", 10)).place(x = 752+70, y = 1000-10-5)

    Button(text = "Wp_upload", command = waypoint, width = 7, font=("Times New Roman", 10)).place(x = 460+70+80+15-5, y =1000+15)

    #Button(text = "ROI_offset", command = ROI_offset, width = 7, font=("Times New Roman", 10)).place(x = 460+70+80+15, y =1000+15)

    clat_entry = Entry()
    clat_entry.place(x = 550+70+80+20-15, y = 985, width = 80, height = 30)
    clon_entry = Entry()
    clon_entry.place(x = 550+70+80+20-15, y = 1015, width = 80, height = 30)

   


    Button(text = "HD :", command = heading, width = 2, font=("Times New Roman", 13)).place(x = 700, y = 950)
    headingset_entry = Entry()
    headingset_entry.place(x = 755, y = 955, width = 30, height = 30)


    Button(text = "AS :", command = speed, width = 2, font=("Times New Roman", 13)).place(x = 795, y = 950)
    speedset_entry = Entry()
    speedset_entry.place(x = 845, y = 955, width = 30, height = 30)


    ###C_Radius = Button(text = "C_Rads", command = C_Radius, width = 5, font=("Times New Roman", 13)).place(x = 638+70, y = 1000)

    ##cradius_entry = Entry()
    #cradius_entry.place(x = 718+70, y = 1000, width = 30, height = 30)

    #Button(text = "Wp_upload", command = waypoint, width = 8, font=("Times New Roman", 10)).place(x = 752+70, y = 1000-10-5)

    Button(text = "ROI_M", command = roi_mode, width = 4, font=("Times New Roman", 10)).place(x = 752+70-30, y = 1000+10+5)

    data311_ = Label(text = "R=", font=("Times New Roman", 9)).place(x = 752+70-30+60, y = 1000+10+5+1)
    data311_ = Label(text = "(m)", font=("Times New Roman", 7)).place(x = 752+70-30+60, y = 1000+10+5+15)
    ROI_R_entry = Entry()
    ROI_R_entry.place(x = 752+70-30+60+20, y = 1000+10+5, width = 30, height = 30)

    data311_ = Label(text = "S=", font=("Times New Roman", 9)).place(x = 752+70-30+60+20+25+5, y = 1000+10+5+1)
    data311_ = Label(text = "(m/s)", font=("Times New Roman", 7)).place(x = 752+70-30+60+20+25+5, y = 1000+10+5+15)
    ROI_S_entry = Entry()
    ROI_S_entry.place(x = 752+70-30+60+20+45+5, y = 1000+10+5, width = 30, height = 30)

    ##data313_ = Label(text = "No.of.uavs =", font=("Times New Roman", 15)).place(x = 880, y= 1000)
    ##no_of_uavs_entry = Entry()
    ##no_of_uavs_entry.place(x = 990, y = 1000, width = 50, height = 30)

    Button(text = "SKIP_WP:", command = skip, width = 6, font=("Times New Roman", 9)).place(x = 860+70+15+10, y = 985)
    skip1set_entry = Entry()
    skip1set_entry.place(x = 960+70-5, y = 985, width = 30, height = 30)

    Button(text = "SKIP_WP2:", command = skip2, width = 6, font=("Times New Roman", 9)).place(x = 860+70+15+10, y = 1015)
    skip2set_entry = Entry()
    skip2set_entry.place(x = 960+70-5, y = 1015, width = 30, height = 30)

    Button(text = "GUD_M", command = guided_main, width = 4, font=("Times New Roman", 10)).place(x = 960+70+30, y = 985)
    Button(text = "GUD_S", command = guided_sec, width = 4, font=("Times New Roman", 10)).place(x = 960+70+30, y = 1015)

    Button(text = "send.txt", command = send_mission_to_uav_all, width = 4, font=("Times New Roman", 10)).place(x = 1000+70+50, y = 1000-10-5)
    Button(text = "No.uavs", command = no_uavs, width = 3,  font=("Times New Roman", 10)).place(x = 1000+70+50+15+10+20+10+3, y = 1000-10-5)
    no_uavsset_entry = Entry()
    no_uavsset_entry.place(x = 1080+70+50+15+10+10-2, y = 1000-10, width = 30, height = 25)
    '''
    Button(text = "MASTER =", command = master, width = 7, font=("Times New Roman", 10)).place(x = 1120+70+50+15+10, y = 1000-10-2)
    masterset_entry = Entry()
    masterset_entry.place(x = 1230+70+50-10+10, y = 1000-10, width = 30, height = 25)	
    mode_button1=Button(text="Set", command=mode_button1, font=("Times New Roman", 15)).place(x = 860, y = 40,height=30, width = 40)  
    mode_button2=Button(text="Set", command=mode_button2, font=("Times New Roman", 15)).place(x=860,y=75,height=30, width = 40)
    mode_button3=Button(text="Set", command=mode_button3, font=("Times New Roman", 15)).place(x=860,y=110,height=30, width = 40)   
    mode_button4=Button(text="Set", command=mode_button4, font=("Times New Roman", 15)).place(x=860,y=145,height=30, width = 40)
    mode_button5=Button(text="Set", command=mode_button5, font=("Times New Roman", 15)).place(x=860,y=180,height=30, width = 40)  
    mode_button6=Button(text="Set", command=mode_button6, font=("Times New Roman", 15)).place(x=860,y=215,height=30, width = 40)
    mode_button7=Button(text="Set", command=mode_button7, font=("Times New Roman", 15)).place(x=860,y=250,height=30, width = 40)   
    mode_button8=Button(text="Set", command=mode_button8, font=("Times New Roman", 15)).place(x=860,y=285,height=30, width = 40)
    mode_button9=Button(text="Set", command=mode_button9, font=("Times New Roman", 15)).place(x=860,y=320,height=30, width = 40)   
    mode_button10=Button(text="Set", command=mode_button10, font=("Times New Roman", 15)).place(x=860,y=355,height=30, width = 40) 
    mode_button11=Button(text="Set", command=mode_button11, font=("Times New Roman", 15)).place(x=860,y=390,height=30, width = 40)     
    mode_button12=Button(text="Set", command=mode_button12, font=("Times New Roman", 15)).place(x=860,y=425,height=30, width = 40)
    mode_button13=Button(text="Set", command=mode_button13, font=("Times New Roman", 15)).place(x=860,y=460,height=30, width = 40)    
    mode_button14=Button(text="Set", command=mode_button14, font=("Times New Roman", 15)).place(x=860,y=495,height=30, width = 40)
    mode_button15=Button(text="Set", command=mode_button15, font=("Times New Roman", 15)).place(x=860,y=530,height=30, width = 40)		
    '''
    Button(text = "save_mission_all", command = save_mission_to_uav_all, width = 10, font=("Times New Roman", 10)).place(x = 1000+70+50, y = 1000+15+2)
    #Home_pos_1 = Button(text = "Home_pos", command = Home_pos, width = 7, font=("Times New Roman", 10)).place(x = 1120+70+20+10, y = 1000+15+2)
    Home_pos_1 = Button(text = "Home_pos_move", command = Home_pos, width = 9, font=("Times New Roman", 10)).place(x = 1120+70+20+10, y = 1000+15+2)
    Target_payload = Button(text = "home_lock", command = home_lock, width = 7, font=("Times New Roman", 10)).place(x = 1180+70+50+10, y = 1000+15+2)


    data311_ = Label(text = "Map_zoom(17)", font=("Times New Roman", 9)).place(x = 1700+20+100-20, y= 880+60)
    #..data311_ = Label(text = "1-uav(800)", font=("Times New Roman", 8)).place(x = 1700+20+100-20+10, y= 880+60+15)
    scaleset_entry = Entry()
    scaleset_entry.place(x = 1700+20+100+60-20+15, y= 880+60, width = 30, height = 25)	
    #Target_payload = Button(text = "pause_mission", command = pause_mission, width = 8, font=("Times New Roman", 10)).place(x = 1700+20+100, y= 880+60)
    
    Target_payload = Button(text = "search_mission", command = generate_search_misison, width = 8, font=("Times New Roman", 8)).place(x = 1700+20+100, y= 880+80+10)
    
    data311_ = Label(text = "grid_line_space", font=("Times New Roman", 8)).place(x = 1700+20+90, y= 880+100+20)
    #data311 = Label(text = "search_area", font=("Times New Roman", 9)).place(1700+20+100, y= 880+100+10)
    search_aera_set_entry = Entry()
    search_aera_set_entry.place(x=1700+20+150, y= 880+100+20, width = 30, height = 25)
    
    data311_ = Label(text = "angle_rotation", font=("Times New Roman", 9)).place(x = 1700+20+90, y= 880+120+20)
    search_count_set_entry = Entry()
    search_count_set_entry.place(x = 1700+20+150, y= 880+120+20, width = 30, height = 25)

    varCheckButton3 = tkinter.IntVar()
    tkCheckButton = tkinter.Checkbutton(
    root,
    text="self_heal_alt",
    variable=varCheckButton3,
    command=setCheckButtonText3, font=("Times New Roman", 10)).place(x = 1180+70+100-150, y= 1000+15+35+2)

    varCheckButton4 = tkinter.IntVar()
    tkCheckButton = tkinter.Checkbutton(
    root,
    text="Drop_all_one_point",
    variable=varCheckButton4,
    command=setCheckButtonText4, font=("Times New Roman", 10)).place(x = 1180+70+100-350, y= 1000+15+35+2)

    varCheckButton5 = tkinter.IntVar()
    tkCheckButton = tkinter.Checkbutton(
    root,
    text="payload_drop_enable",
    variable=varCheckButton5,
    command=setCheckButtonText5, font=("Times New Roman", 10)).place(x = 1180+70+100-500, y= 1000+15+35+2)
    '''

    #Button(text = "Air_break", command = Air_break, width = 8, font=("Times New Roman", 8)).place(x = 1180+70+100-600, y= 1000+15+35+2)

    #Button(text = "Move_payload", command = move_payload, width = 8, font=("Times New Roman", 8)).place(x = 1180+70+100-700, y= 1000+15+35+2)
    
    
    #Button(text = "THD", command = Target_Heading_compute, width = 8, font=("Times New Roman", 8)).place(x = 1180+70+100-1000, y= 1000+15+35+2)


    #Target_payload = Button(text = "pause_mission", command = pause_mission, width = 8, font=("Times New Roman", 10)).place(x = 1180+70+100-40, y= 1000+15+35)

    #Button(text = "start_F_Guided", command = send, width = 8, font=("Times New Roman", 8)).place(x = 1180+70+150, y = 1000+15+35)
    #Button(text = "start_R_Guided", command = rece, width = 8, font=("Times New Roman", 8)).place(x = 1180+70+200+30, y = 1000+15+35)

    #Target_payload = Button(text = "stop_forward", command = stop_forward, width = 7, font=("Times New Roman", 10)).place(x = 1180+70+300+10, y = 1000+15+35)
    #Target_payload = Button(text = "stop_return", command = stop_return, width = 7, font=("Times New Roman", 10)).place(x = 1180+70+380+10, y = 1000+15+35)
    #Target_payload = Button(text = "download_mission_Guided", command = download_mission_Guided, width = 20, font=("Times New Roman", 10)).place(x = 1180+70+470, y = 1000+15+35)

    

    """
    ##data313_.config(font=("Courier", 44))
    varCheckButton = tkinter.IntVar()
    tkCheckButton = tkinter.Checkbutton(
    root,
    text="ON_CAM",
    variable=varCheckButton,
    command=setCheckButtonText, font=("Times New Roman", 15)).place(x = 780, y = 20)


    varCheckButton1 = tkinter.IntVar()
    tkCheckButton = tkinter.Checkbutton(
    root,
    text="cam1",
    variable=varCheckButton1,
    command=setCheckButtonText1, font=("Times New Roman", 15)).place(x = 820, y = 20)

    varCheckButton2 = tkinter.IntVar()
    tkinter.Checkbutton(
    root,
    text="cam2",
    variable=varCheckButton2,
    command=setCheckButtonText2, font=("Times New Roman", 15)).place(x = 880, y = 20)

    """
    #t3 = threading.Thread(target = update_ip_link_status)
    #t3.daemon = True
    #t3.start() 
    """
      
    ''' 
    mode_button16=Button(text="Set", command=mode_button16, font=("Times New Roman", 15)).place(x=960,y=565,height=30, width = 40)
    mode_button17=Button(text="Set", command=mode_button17, font=("Times New Roman", 15)).place(x=960,y=600,height=30, width = 40)
    mode_button18=Button(text="Set", command=mode_button18, font=("Times New Roman", 15)).place(x=960,y=635,height=30, width = 40)
    mode_button19=Button(text="Set", command=mode_button19, font=("Times New Roman", 15)).place(x=960,y=670,height=30, width = 40)
    mode_button20=Button(text="Set", command=mode_button20, font=("Times New Roman", 15)).place(x=960,y=705,height=30, width = 40)
    mode_button21=Button(text="Set", command=mode_button21, font=("Times New Roman", 15)).place(x=960,y=740,height=30, width = 40)
    mode_button22=Button(text="Set", command=mode_button22, font=("Times New Roman", 15)).place(x=960,y=775,height=30, width = 40)
    mode_button23=Button(text="Set", command=mode_button23, font=("Times New Roman", 15)).place(x=960,y=810,height=30, width = 40)
    mode_button24=Button(text="Set", command=mode_button24, font=("Times New Roman", 15)).place(x=960,y=845,height=30, width = 40)
    mode_button25=Button(text="Set", command=mode_button25, font=("Times New Roman", 15)).place(x=960,y=880,height=30, width = 40)
   
    checkboxvalue_01 = IntVar()
    Checkbutton(root, variable=checkboxvalue_01).place(x = 1000, y = 40, width=30, height=30)
    checkboxvalue_02 = IntVar()
    Checkbutton(root, variable=checkboxvalue_02).place(x = 1000, y = 75, width=30, height=30)
    checkboxvalue_03 = IntVar()
    Checkbutton(root, variable=checkboxvalue_03).place(x = 1000, y = 110, width=30, height=30)
    checkboxvalue_04 = IntVar()
    Checkbutton(root, variable=checkboxvalue_04).place(x = 1000, y = 145, width=30, height=30)
    checkboxvalue_05 = IntVar()
    Checkbutton(root, variable=checkboxvalue_05).place(x = 1000, y = 180, width=30, height=30)
    checkboxvalue_06 = IntVar()
    Checkbutton(root, variable=checkboxvalue_06).place(x = 1000, y = 215, width=30, height=30)
    checkboxvalue_07 = IntVar()
    Checkbutton(root, variable=checkboxvalue_07).place(x = 1000, y = 250, width=30, height=30)
    checkboxvalue_08 = IntVar()
    Checkbutton(root, variable=checkboxvalue_08).place(x = 1000, y = 285, width=30, height=30)
    checkboxvalue_09 = IntVar()
    Checkbutton(root, variable=checkboxvalue_09).place(x = 1000, y = 320, width=30, height=30)
    checkboxvalue_010 = IntVar()
    Checkbutton(root, variable=checkboxvalue_010).place(x = 1000, y = 355, width=30, height=30)
    checkboxvalue_011 = IntVar()
    Checkbutton(root, variable=checkboxvalue_011).place(x = 1000, y = 390, width=30, height=30)
    checkboxvalue_012 = IntVar()
    Checkbutton(root, variable=checkboxvalue_012).place(x = 1000, y = 425, width=30, height=30)
    checkboxvalue_013 = IntVar()
    Checkbutton(root, variable=checkboxvalue_013).place(x = 1000, y = 460, width=30, height=30)
    checkboxvalue_014 = IntVar()
    Checkbutton(root, variable=checkboxvalue_014).place(x = 1000, y = 495, width=30, height=30)
    checkboxvalue_015 = IntVar()
    
    Checkbutton(root, variable=checkboxvalue_015).place(x = 1000, y = 530, width=30, height=30)
    checkboxvalue_016 = IntVar()
    Checkbutton(root, variable=checkboxvalue_016).place(x = 1000, y = 565, width=30, height=30)
    checkboxvalue_017 = IntVar()
    Checkbutton(root, variable=checkboxvalue_017).place(x = 1000, y = 600, width=30, height=30)
    checkboxvalue_018 = IntVar()
    Checkbutton(root, variable=checkboxvalue_018).place(x = 1000, y = 635, width=30, height=30)
    checkboxvalue_019 = IntVar()
    Checkbutton(root, variable=checkboxvalue_019).place(x = 1000, y = 670, width=30, height=30)
    checkboxvalue_020 = IntVar()
    Checkbutton(root, variable=checkboxvalue_020).place(x = 1000, y = 705, width=30, height=30)
    checkboxvalue_021 = IntVar()
    Checkbutton(root, variable=checkboxvalue_021).place(x = 1000, y = 740, width=30, height=30)
    checkboxvalue_022 = IntVar()
    Checkbutton(root, variable=checkboxvalue_022).place(x = 1000, y = 775, width=30, height=30)
    checkboxvalue_023 = IntVar()
    Checkbutton(root, variable=checkboxvalue_023).place(x = 1000, y = 810, width=30, height=30)
    checkboxvalue_024 = IntVar()
    Checkbutton(root, variable=checkboxvalue_024).place(x = 1000, y = 845, width=30, height=30)
    checkboxvalue_025 = IntVar()
    Checkbutton(root, variable=checkboxvalue_025).place(x = 1000, y = 880, width=30, height=30)
    '''

    Button(text="Go", command=mode_button_1, font=("Times New Roman", 15)).place(x=1210,y=40,height=30, width = 35)  
    Button(text="Go", command=mode_button_2, font=("Times New Roman", 15)).place(x=1210,y=75,height=30, width = 35)
    Button(text="Go", command=mode_button_3, font=("Times New Roman", 15)).place(x=1210,y=110,height=30, width = 35)   
    Button(text="Go", command=mode_button_4, font=("Times New Roman", 15)).place(x=1210,y=145,height=30, width = 35)
    Button(text="Go", command=mode_button_5, font=("Times New Roman", 15)).place(x=1210,y=180,height=30, width = 35)  
    Button(text="Go", command=mode_button_6, font=("Times New Roman", 15)).place(x=1210,y=215,height=30, width = 35)
    Button(text="Go", command=mode_button_7, font=("Times New Roman", 15)).place(x=1210,y=250,height=30, width = 35)   
    Button(text="Go", command=mode_button_8, font=("Times New Roman", 15)).place(x=1210,y=285,height=30, width = 35)
    Button(text="Go", command=mode_button_9, font=("Times New Roman", 15)).place(x=1210,y=320,height=30, width = 35)   
    Button(text="Go", command=mode_button_10, font=("Times New Roman", 15)).place(x=1210,y=355,height=30, width = 35) 
    Button(text="Go", command=mode_button_11, font=("Times New Roman", 15)).place(x=1210,y=390,height=30, width = 35)     
    Button(text="Go", command=mode_button_12, font=("Times New Roman", 15)).place(x=1210,y=425,height=30, width = 35)
    Button(text="Go", command=mode_button_13, font=("Times New Roman", 15)).place(x=1210,y=460,height=30, width = 35)    
    Button(text="Go", command=mode_button_14, font=("Times New Roman", 15)).place(x=1210,y=495,height=30, width = 35)
    Button(text="Go", command=mode_button_15, font=("Times New Roman", 15)).place(x=1210,y=530,height=30, width = 35) 
    '''  
    Button(text="Go", command=mode_button_16, font=("Times New Roman", 15)).place(x=1210,y=565,height=30, width = 35)
    Button(text="Go", command=mode_button_17, font=("Times New Roman", 15)).place(x=1210,y=600,height=30, width = 35)
    Button(text="Go", command=mode_button_18, font=("Times New Roman", 15)).place(x=1210,y=635,height=30, width = 35)
    Button(text="Go", command=mode_button_19, font=("Times New Roman", 15)).place(x=1210,y=670,height=30, width = 35)
    Button(text="Go", command=mode_button_20, font=("Times New Roman", 15)).place(x=1210,y=705,height=30, width = 35)
    Button(text="Go", command=mode_button_21, font=("Times New Roman", 15)).place(x=1210,y=740,height=30, width = 35)
    Button(text="Go", command=mode_button_22, font=("Times New Roman", 15)).place(x=1210,y=775,height=30, width = 35)
    Button(text="Go", command=mode_button_23, font=("Times New Roman", 15)).place(x=1210,y=810,height=30, width = 35)
    Button(text="Go", command=mode_button_24, font=("Times New Roman", 15)).place(x=1210,y=845,height=30, width = 35)
    Button(text="Go", command=mode_button_25, font=("Times New Roman", 15)).place(x=1210,y=880,height=30, width = 35)
    '''
    checkboxvalue_1 = IntVar()
    Checkbutton(root, variable=checkboxvalue_1).place(x = 1250, y = 40, width=30, height=30)
    checkboxvalue_2 = IntVar()
    Checkbutton(root, variable=checkboxvalue_2).place(x = 1250, y = 75, width=30, height=30)
    checkboxvalue_3 = IntVar()
    Checkbutton(root, variable=checkboxvalue_3).place(x = 1250, y = 110, width=30, height=30)
    checkboxvalue_4 = IntVar()
    Checkbutton(root, variable=checkboxvalue_4).place(x = 1250, y = 145, width=30, height=30)
    checkboxvalue_5 = IntVar()
    Checkbutton(root, variable=checkboxvalue_5).place(x = 1250, y = 180, width=30, height=30)
    checkboxvalue_6 = IntVar()
    Checkbutton(root, variable=checkboxvalue_6).place(x = 1250, y = 215, width=30, height=30)
    checkboxvalue_7 = IntVar()
    Checkbutton(root, variable=checkboxvalue_7).place(x = 1250, y = 250, width=30, height=30)
    checkboxvalue_8 = IntVar()
    Checkbutton(root, variable=checkboxvalue_8).place(x = 1250, y = 285, width=30, height=30)
    checkboxvalue_9 = IntVar()
    Checkbutton(root, variable=checkboxvalue_9).place(x = 1250, y = 320, width=30, height=30)
    checkboxvalue_10 = IntVar()
    Checkbutton(root, variable=checkboxvalue_10).place(x = 1250, y = 355, width=30, height=30)
    checkboxvalue_11 = IntVar()
    Checkbutton(root, variable=checkboxvalue_11).place(x = 1250, y = 390, width=30, height=30)
    checkboxvalue_12 = IntVar()
    Checkbutton(root, variable=checkboxvalue_12).place(x = 1250, y = 425, width=30, height=30)
    checkboxvalue_13 = IntVar()
    Checkbutton(root, variable=checkboxvalue_13).place(x = 1250, y = 460, width=30, height=30)
    checkboxvalue_14 = IntVar()
    Checkbutton(root, variable=checkboxvalue_14).place(x = 1250, y = 495, width=30, height=30)
    checkboxvalue_15 = IntVar()
    Checkbutton(root, variable=checkboxvalue_15).place(x = 1250, y = 530, width=30, height=30)
    checkboxvalue_16 = IntVar()
    Checkbutton(root, variable=checkboxvalue_16).place(x = 1250, y = 565, width=30, height=30)
    checkboxvalue_17 = IntVar()
    Checkbutton(root, variable=checkboxvalue_17).place(x = 1250, y = 600, width=30, height=30)
    checkboxvalue_18 = IntVar()
    Checkbutton(root, variable=checkboxvalue_18).place(x = 1250, y = 635, width=30, height=30)
    checkboxvalue_19 = IntVar()
    Checkbutton(root, variable=checkboxvalue_19).place(x = 1250, y = 670, width=30, height=30)
    checkboxvalue_20 = IntVar()
    Checkbutton(root, variable=checkboxvalue_20).place(x = 1250, y = 705, width=30, height=30)
    checkboxvalue_21 = IntVar()
    Checkbutton(root, variable=checkboxvalue_21).place(x = 1250, y = 740, width=30, height=30)
    checkboxvalue_22 = IntVar()
    Checkbutton(root, variable=checkboxvalue_22).place(x = 1250, y = 775, width=30, height=30)
    checkboxvalue_23 = IntVar()
    Checkbutton(root, variable=checkboxvalue_23).place(x = 1250, y = 810, width=30, height=30)
    checkboxvalue_24 = IntVar()
    Checkbutton(root, variable=checkboxvalue_24).place(x = 1250, y = 845, width=30, height=30)
    checkboxvalue_25 = IntVar()
    Checkbutton(root, variable=checkboxvalue_25).place(x = 1250, y = 880, width=30, height=30)



    goto_button1=Button(text="GO", command=goto_button1, font=("Times New Roman", 15)).place(x=1470,y=40,height=30, width = 45)   
    goto_button2=Button(text="GO", command=goto_button2, font=("Times New Roman", 15)).place(x=1470, y=75,height=30, width = 45)
    goto_button3=Button(text="GO", command=goto_button3, font=("Times New Roman", 15)).place(x=1470,y=110,height=30, width = 45)   
    goto_button4=Button(text="GO", command=goto_button4, font=("Times New Roman", 15)).place(x=1470,y=145,height=30, width = 45)
    goto_button5=Button(text="GO", command=goto_button5, font=("Times New Roman", 15)).place(x=1470,y=180,height=30, width = 45)  
    goto_button6=Button(text="GO", command=goto_button6, font=("Times New Roman", 15)).place(x=1470,y=215,height=30, width = 45)
    goto_button7=Button(text="GO", command=goto_button7, font=("Times New Roman", 15)).place(x=1470,y=250,height=30, width = 45)   
    goto_button8=Button(text="GO", command=goto_button8, font=("Times New Roman", 15)).place(x=1470,y=285,height=30, width = 45)
    goto_button9=Button(text="GO", command=goto_button9, font=("Times New Roman", 15)).place(x=1470,y=320,height=30, width = 45)   
    goto_button10=Button(text="GO", command=goto_button10, font=("Times New Roman", 15)).place(x=1470,y=355,height=30, width = 45) 
    goto_button11=Button(text="GO", command=goto_button11, font=("Times New Roman", 15)).place(x=1470,y=390,height=30, width = 45)     
    goto_button12=Button(text="GO", command=goto_button12, font=("Times New Roman", 15)).place(x=1470,y=425,height=30, width = 45)
    goto_button13=Button(text="GO", command=goto_button13, font=("Times New Roman", 15)).place(x=1470,y=460,height=30, width = 45)    
    goto_button14=Button(text="GO", command=goto_button14, font=("Times New Roman", 15)).place(x=1470,y=495,height=30, width = 45)
    goto_button15=Button(text="GO", command=goto_button15, font=("Times New Roman", 15)).place(x=1470,y=530,height=30, width = 45)   
    goto_button16=Button(text="GO", command=goto_button16, font=("Times New Roman", 15)).place(x=1470,y=565,height=30, width = 45)
    goto_button17=Button(text="GO", command=goto_button17, font=("Times New Roman", 15)).place(x=1470,y=600,height=30, width = 45)
    goto_button18=Button(text="GO", command=goto_button18, font=("Times New Roman", 15)).place(x=1470,y=635,height=30, width = 45)
    goto_button19=Button(text="GO", command=goto_button19, font=("Times New Roman", 15)).place(x=1470,y=670,height=30, width = 45)
    goto_button20=Button(text="GO", command=goto_button20, font=("Times New Roman", 15)).place(x=1470,y=705,height=30, width = 45)
    goto_button21=Button(text="GO", command=goto_button21, font=("Times New Roman", 15)).place(x=1470,y=740,height=30, width = 45)
    goto_button22=Button(text="GO", command=goto_button22, font=("Times New Roman", 15)).place(x=1470,y=775,height=30, width = 45)
    goto_button23=Button(text="GO", command=goto_button23, font=("Times New Roman", 15)).place(x=1470,y=810,height=30, width = 45)
    goto_button24=Button(text="GO", command=goto_button24, font=("Times New Roman", 15)).place(x=1470,y=845,height=30, width = 45)
    goto_button25=Button(text="GO", command=goto_button25, font=("Times New Roman", 15)).place(x=1470,y=880,height=30, width = 45)

    alt_button1=Button(text="GO", command=alt_button1, font=("Times New Roman", 15)).place(x=1560,y=40,height=30, width = 45)     
    alt_button2=Button(text="GO", command=alt_button2, font=("Times New Roman", 15)).place(x=1560, y=75,height=30, width = 45)
    alt_button3=Button(text="GO", command=alt_button3, font=("Times New Roman", 15)).place(x=1560,y=110,height=30, width = 45)   
    alt_button4=Button(text="GO", command=alt_button4, font=("Times New Roman", 15)).place(x=1560,y=145,height=30, width = 45)
    alt_button5=Button(text="GO", command=alt_button5, font=("Times New Roman", 15)).place(x=1560,y=180,height=30, width = 45)  
    alt_button6=Button(text="GO", command=alt_button6, font=("Times New Roman", 15)).place(x=1560,y=215,height=30, width = 45)
    alt_button7=Button(text="GO", command=alt_button7, font=("Times New Roman", 15)).place(x=1560,y=250,height=30, width = 45)   
    alt_button8=Button(text="GO", command=alt_button8, font=("Times New Roman", 15)).place(x=1560,y=285,height=30, width = 45)
    alt_button9=Button(text="GO", command=alt_button9, font=("Times New Roman", 15)).place(x=1560,y=320,height=30, width = 45)   
    alt_button10=Button(text="GO", command=alt_button10, font=("Times New Roman", 15)).place(x=1560,y=355,height=30, width = 45) 
    alt_button11=Button(text="GO", command=alt_button11, font=("Times New Roman", 15)).place(x=1560,y=390,height=30, width = 45)     
    alt_button12=Button(text="GO", command=alt_button12, font=("Times New Roman", 15)).place(x=1560,y=425,height=30, width = 45)
    alt_button13=Button(text="GO", command=alt_button13, font=("Times New Roman", 15)).place(x=1560,y=460,height=30, width = 45)    
    alt_button14=Button(text="GO", command=alt_button14, font=("Times New Roman", 15)).place(x=1560,y=495,height=30, width = 45)
    alt_button15=Button(text="GO", command=alt_button15, font=("Times New Roman", 15)).place(x=1560,y=530,height=30, width = 45)   
    alt_button16=Button(text="GO", command=alt_button16, font=("Times New Roman", 15)).place(x=1560,y=565,height=30, width = 45)
    alt_button17=Button(text="GO", command=alt_button17, font=("Times New Roman", 15)).place(x=1560,y=600,height=30, width = 45)
    alt_button18=Button(text="GO", command=alt_button18, font=("Times New Roman", 15)).place(x=1560,y=635,height=30, width = 45)
    alt_button19=Button(text="GO", command=alt_button19, font=("Times New Roman", 15)).place(x=1560,y=670,height=30, width = 45)
    alt_button20=Button(text="GO", command=alt_button20, font=("Times New Roman", 15)).place(x=1560,y=705,height=30, width = 45)
    alt_button21=Button(text="GO", command=alt_button21, font=("Times New Roman", 15)).place(x=1560,y=740,height=30, width = 45)
    alt_button22=Button(text="GO", command=alt_button22, font=("Times New Roman", 15)).place(x=1560,y=775,height=30, width = 45)
    alt_button23=Button(text="GO", command=alt_button23, font=("Times New Roman", 15)).place(x=1560,y=810,height=30, width = 45)
    alt_button24=Button(text="GO", command=alt_button24, font=("Times New Roman", 15)).place(x=1560,y=845,height=30, width = 45)
    alt_button25=Button(text="GO", command=alt_button25, font=("Times New Roman", 15)).place(x=1560,y=880,height=30, width = 45)


    payload_button1=Button(text="DROP", command=payload_button1, font=("Times New Roman", 15)).place(x = 1610, y = 40,height=30, width = 65)      
    payload_button2=Button(text="DROP", command=payload_button2, font=("Times New Roman", 15)).place(x=1610,y=75,height=30, width = 65)
    payload_button3=Button(text="DROP", command=payload_button3, font=("Times New Roman", 15)).place(x=1610,y=110,height=30, width = 60)   
    payload_button4=Button(text="DROP", command=payload_button4, font=("Times New Roman", 15)).place(x=1610,y=145,height=30, width = 60)
    payload_button5=Button(text="DROP", command=payload_button5, font=("Times New Roman", 15)).place(x=1610,y=180,height=30, width = 60)  
    payload_button6=Button(text="DROP", command=payload_button6, font=("Times New Roman", 15)).place(x=1610,y=215,height=30, width = 60)
    payload_button7=Button(text="DROP", command=payload_button7, font=("Times New Roman", 15)).place(x=1610,y=250,height=30, width = 60)   
    payload_button8=Button(text="DROP", command=payload_button8, font=("Times New Roman", 15)).place(x=1610,y=285,height=30, width = 60)
    payload_button9=Button(text="DROP", command=payload_button9, font=("Times New Roman", 15)).place(x=1610,y=320,height=30, width = 60)   
    payload_button10=Button(text="DROP", command=payload_button10, font=("Times New Roman", 15)).place(x=1610,y=355,height=30, width = 60) 
    payload_button11=Button(text="DROP", command=payload_button11, font=("Times New Roman", 15)).place(x=1610,y=390,height=30, width = 60)     
    payload_button12=Button(text="DROP", command=payload_button12, font=("Times New Roman", 15)).place(x=1610,y=425,height=30, width = 60)
    payload_button13=Button(text="DROP", command=payload_button13, font=("Times New Roman", 15)).place(x=1610,y=460,height=30, width = 60)    
    payload_button14=Button(text="DROP", command=payload_button14, font=("Times New Roman", 15)).place(x=1610,y=495,height=30, width = 60)
    payload_button15=Button(text="DROP", command=payload_button15, font=("Times New Roman", 15)).place(x=1610,y=530,height=30, width = 60)   
    payload_button16=Button(text="DROP", command=payload_button16, font=("Times New Roman", 15)).place(x=1610,y=565,height=30, width = 60)
    payload_button17=Button(text="DROP", command=payload_button17, font=("Times New Roman", 15)).place(x=1610,y=600,height=30, width = 60)
    payload_button18=Button(text="DROP", command=payload_button18, font=("Times New Roman", 15)).place(x=1610,y=635,height=30, width = 60)
    payload_button19=Button(text="DROP", command=payload_button19, font=("Times New Roman", 15)).place(x=1610,y=670,height=30, width = 60)
    payload_button20=Button(text="DROP", command=payload_button20, font=("Times New Roman", 15)).place(x=1610,y=705,height=30, width = 60)
    payload_button21=Button(text="DROP", command=payload_button21, font=("Times New Roman", 15)).place(x=1610,y=740,height=30, width = 60)
    payload_button22=Button(text="DROP", command=payload_button22, font=("Times New Roman", 15)).place(x=1610,y=775,height=30, width = 60)
    payload_button23=Button(text="DROP", command=payload_button23, font=("Times New Roman", 15)).place(x=1610,y=810,height=30, width = 60)
    payload_button24=Button(text="DROP", command=payload_button24, font=("Times New Roman", 15)).place(x=1610,y=845,height=30, width = 60)
    payload_button25=Button(text="DROP", command=payload_button25, font=("Times New Roman", 15)).place(x=1610,y=880,height=30, width = 60)

    #..............group 1...............................
    checkboxvalue_G1_1 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G1_1).place(x = 1690+40-5, y = 40, width=30, height=30)
    checkboxvalue_G1_2 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G1_2).place(x = 1690+40-5, y = 75, width=30, height=30)
    checkboxvalue_G1_3 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G1_3).place(x = 1690+40-5, y = 110, width=30, height=30)
    checkboxvalue_G1_4 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G1_4).place(x = 1690+40-5, y = 145, width=30, height=30)
    checkboxvalue_G1_5 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G1_5).place(x = 1690+40-5, y = 180, width=30, height=30)
    checkboxvalue_G1_6 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G1_6).place(x = 1690+40-5, y = 215, width=30, height=30)
    checkboxvalue_G1_7 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G1_7).place(x = 1690+40-5, y = 250, width=30, height=30)
    checkboxvalue_G1_8 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G1_8).place(x = 1690+40-5, y = 285, width=30, height=30)
    checkboxvalue_G1_9 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G1_9).place(x = 1690+40-5, y = 320, width=30, height=30)
    checkboxvalue_G1_10 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G1_10).place(x = 1690+40-5, y = 355, width=30, height=30)
    checkboxvalue_G1_11 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G1_11).place(x = 1690+40-5, y = 390, width=30, height=30)
    checkboxvalue_G1_12 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G1_12).place(x = 1690+40-5, y = 425, width=30, height=30)
    checkboxvalue_G1_13 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G1_13).place(x = 1690+40-5, y = 460, width=30, height=30)
    checkboxvalue_G1_14 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G1_14).place(x = 1690+40-5, y = 495, width=30, height=30)
    checkboxvalue_G1_15 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G1_15).place(x = 1690+40-5, y = 530, width=30, height=30)
    checkboxvalue_G1_16 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G1_16).place(x = 1690+40-5, y = 565, width=30, height=30)
    checkboxvalue_G1_17 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G1_17).place(x = 1690+40-5, y = 600, width=30, height=30)
    checkboxvalue_G1_18 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G1_18).place(x = 1690+40-5, y = 635, width=30, height=30)
    checkboxvalue_G1_19 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G1_19).place(x = 1690+40-5, y = 670, width=30, height=30)
    checkboxvalue_G1_20 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G1_20).place(x = 1690+40-5, y = 705, width=30, height=30)
    checkboxvalue_G1_21 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G1_21).place(x = 1690+40-5, y = 740, width=30, height=30)
    checkboxvalue_G1_22 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G1_22).place(x = 1690+40-5, y = 775, width=30, height=30)
    checkboxvalue_G1_23 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G1_23).place(x = 1690+40-5, y = 810, width=30, height=30)
    checkboxvalue_G1_24 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G1_24).place(x = 1690+40-5, y = 845, width=30, height=30)
    checkboxvalue_G1_25 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G1_25).place(x = 1690+40-5, y = 880, width=30, height=30)

    #........................group 2.............................
    checkboxvalue_G2_1 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G2_1).place(x = 1690+80-10, y = 40, width=30, height=30)
    checkboxvalue_G2_2 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G2_2).place(x = 1690+80-10, y = 75, width=30, height=30)
    checkboxvalue_G2_3 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G2_3).place(x = 1690+80-10, y = 110, width=30, height=30)
    checkboxvalue_G2_4 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G2_4).place(x = 1690+80-10, y = 145, width=30, height=30)
    checkboxvalue_G2_5 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G2_5).place(x = 1690+80-10, y = 180, width=30, height=30)
    checkboxvalue_G2_6 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G2_6).place(x = 1690+80-10, y = 215, width=30, height=30)
    checkboxvalue_G2_7 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G2_7).place(x = 1690+80-10, y = 250, width=30, height=30)
    checkboxvalue_G2_8 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G2_8).place(x = 1690+80-10, y = 285, width=30, height=30)
    checkboxvalue_G2_9 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G2_9).place(x = 1690+80-10, y = 320, width=30, height=30)
    checkboxvalue_G2_10 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G2_10).place(x = 1690+80-10, y = 355, width=30, height=30)
    checkboxvalue_G2_11 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G2_11).place(x = 1690+80-10, y = 390, width=30, height=30)
    checkboxvalue_G2_12 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G2_12).place(x = 1690+80-10, y = 425, width=30, height=30)
    checkboxvalue_G2_13 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G2_13).place(x = 1690+80-10, y = 460, width=30, height=30)
    checkboxvalue_G2_14 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G2_14).place(x = 1690+80-10, y = 495, width=30, height=30)
    checkboxvalue_G2_15 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G2_15).place(x = 1690+80-10, y = 530, width=30, height=30)
    checkboxvalue_G2_16 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G2_16).place(x = 1690+80-10, y = 565, width=30, height=30)
    checkboxvalue_G2_17 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G2_17).place(x = 1690+80-10, y = 600, width=30, height=30)
    checkboxvalue_G2_18 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G2_18).place(x = 1690+80-10, y = 635, width=30, height=30)
    checkboxvalue_G2_19 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G2_19).place(x = 1690+80-10, y = 670, width=30, height=30)
    checkboxvalue_G2_20 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G2_20).place(x = 1690+80-10, y = 705, width=30, height=30)
    checkboxvalue_G2_21 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G2_21).place(x = 1690+80-10, y = 740, width=30, height=30)
    checkboxvalue_G2_22 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G2_22).place(x = 1690+80-10, y = 775, width=30, height=30)
    checkboxvalue_G2_23 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G2_23).place(x = 1690+80-10, y = 810, width=30, height=30)
    checkboxvalue_G2_24 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G2_24).place(x = 1690+80-10, y = 845, width=30, height=30)
    checkboxvalue_G2_25 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G2_25).place(x = 1690+80-10, y = 880, width=30, height=30)
    #...................................group 3.........................
    checkboxvalue_G3_1 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G3_1).place(x = 1690+120-15, y = 40, width=30, height=30)
    checkboxvalue_G3_2 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G3_2).place(x = 1690+120-15, y = 75, width=30, height=30)
    checkboxvalue_G3_3 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G3_3).place(x = 1690+120-15, y = 110, width=30, height=30)
    checkboxvalue_G3_4 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G3_4).place(x = 1690+120-15, y = 145, width=30, height=30)
    checkboxvalue_G3_5 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G3_5).place(x = 1690+120-15, y = 180, width=30, height=30)
    checkboxvalue_G3_6 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G3_6).place(x = 1690+120-15, y = 215, width=30, height=30)
    checkboxvalue_G3_7 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G3_7).place(x = 1690+120-15, y = 250, width=30, height=30)
    checkboxvalue_G3_8 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G3_8).place(x = 1690+120-15, y = 285, width=30, height=30)
    checkboxvalue_G3_9 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G3_9).place(x = 1690+120-15, y = 320, width=30, height=30)
    checkboxvalue_G3_10 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G3_10).place(x = 1690+120-15, y = 355, width=30, height=30)
    checkboxvalue_G3_11 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G3_11).place(x = 1690+120-15, y = 390, width=30, height=30)
    checkboxvalue_G3_12 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G3_12).place(x = 1690+120-15, y = 425, width=30, height=30)
    checkboxvalue_G3_13 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G3_13).place(x = 1690+120-15, y = 460, width=30, height=30)
    checkboxvalue_G3_14 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G3_14).place(x = 1690+120-15, y = 495, width=30, height=30)
    checkboxvalue_G3_15 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G3_15).place(x = 1690+120-15, y = 530, width=30, height=30)
    checkboxvalue_G3_16 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G3_16).place(x = 1690+120-15, y = 565, width=30, height=30)
    checkboxvalue_G3_17 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G3_17).place(x = 1690+120-15, y = 600, width=30, height=30)
    checkboxvalue_G3_18 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G3_18).place(x = 1690+120-15, y = 635, width=30, height=30)
    checkboxvalue_G3_19 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G3_19).place(x = 1690+120-15, y = 670, width=30, height=30)
    checkboxvalue_G3_20 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G3_20).place(x = 1690+120-15, y = 705, width=30, height=30)
    checkboxvalue_G3_21 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G3_21).place(x = 1690+120-15, y = 740, width=30, height=30)
    checkboxvalue_G3_22 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G3_22).place(x = 1690+120-15, y = 775, width=30, height=30)
    checkboxvalue_G3_23 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G3_23).place(x = 1690+120-15, y = 810, width=30, height=30)
    checkboxvalue_G3_24 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G3_24).place(x = 1690+120-15, y = 845, width=30, height=30)
    checkboxvalue_G3_25 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G3_25).place(x = 1690+120-15, y = 880, width=30, height=30)
    #.......................group 4.......................
    checkboxvalue_G4_1 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G4_1).place(x = 1690+160-20, y = 40, width=30, height=30)
    checkboxvalue_G4_2 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G4_2).place(x = 1690+160-20, y = 75, width=30, height=30)
    checkboxvalue_G4_3 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G4_3).place(x = 1690+160-20, y = 110, width=30, height=30)
    checkboxvalue_G4_4 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G4_4).place(x = 1690+160-20, y = 145, width=30, height=30)
    checkboxvalue_G4_5 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G4_5).place(x = 1690+160-20, y = 180, width=30, height=30)
    checkboxvalue_G4_6 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G4_6).place(x = 1690+160-20, y = 215, width=30, height=30)
    checkboxvalue_G4_7 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G4_7).place(x = 1690+160-20, y = 250, width=30, height=30)
    checkboxvalue_G4_8 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G4_8).place(x = 1690+160-20, y = 285, width=30, height=30)
    checkboxvalue_G4_9 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G4_9).place(x = 1690+160-20, y = 320, width=30, height=30)
    checkboxvalue_G4_10 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G4_10).place(x = 1690+160-20, y = 355, width=30, height=30)
    checkboxvalue_G4_11 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G4_11).place(x = 1690+160-20, y = 390, width=30, height=30)
    checkboxvalue_G4_12 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G4_12).place(x = 1690+160-20, y = 425, width=30, height=30)
    checkboxvalue_G4_13 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G4_13).place(x = 1690+160-20, y = 460, width=30, height=30)
    checkboxvalue_G4_14 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G4_14).place(x = 1690+160-20, y = 495, width=30, height=30)
    checkboxvalue_G4_15 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G4_15).place(x = 1690+160-20, y = 530, width=30, height=30)
    checkboxvalue_G4_16 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G4_16).place(x = 1690+160-20, y = 565, width=30, height=30)
    checkboxvalue_G4_17 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G4_17).place(x = 1690+160-20, y = 600, width=30, height=30)
    checkboxvalue_G4_18 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G4_18).place(x = 1690+160-20, y = 635, width=30, height=30)
    checkboxvalue_G4_19 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G4_19).place(x = 1690+160-20, y = 670, width=30, height=30)
    checkboxvalue_G4_20 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G4_20).place(x = 1690+160-20, y = 705, width=30, height=30)
    checkboxvalue_G4_21 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G4_21).place(x = 1690+160-20, y = 740, width=30, height=30)
    checkboxvalue_G4_22 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G4_22).place(x = 1690+160-20, y = 775, width=30, height=30)
    checkboxvalue_G4_23 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G4_23).place(x = 1690+160-20, y = 810, width=30, height=30)
    checkboxvalue_G4_24 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G4_24).place(x = 1690+160-20, y = 845, width=30, height=30)
    checkboxvalue_G4_25 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G4_25).place(x = 1690+160-20, y = 880, width=30, height=30)

   
    #.........................group 5.........................
    checkboxvalue_G5_1 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G5_1).place(x = 1690+200-25, y = 40, width=30, height=30)
    checkboxvalue_G5_2 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G5_2).place(x = 1690+200-25, y = 75, width=30, height=30)
    checkboxvalue_G5_3 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G5_3).place(x = 1690+200-25, y = 110, width=30, height=30)
    checkboxvalue_G5_4 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G5_4).place(x = 1690+200-25, y = 145, width=30, height=30)
    checkboxvalue_G5_5 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G5_5).place(x = 1690+200-25, y = 180, width=30, height=30)
    checkboxvalue_G5_6 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G5_6).place(x = 1690+200-25, y = 215, width=30, height=30)
    checkboxvalue_G5_7 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G5_7).place(x = 1690+200-25, y = 250, width=30, height=30)
    checkboxvalue_G5_8 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G5_8).place(x = 1690+200-25, y = 285, width=30, height=30)
    checkboxvalue_G5_9 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G5_9).place(x = 1690+200-25, y = 320, width=30, height=30)
    checkboxvalue_G5_10 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G5_10).place(x = 1690+200-25, y = 355, width=30, height=30)
    checkboxvalue_G5_11 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G5_11).place(x = 1690+200-25, y = 390, width=30, height=30)
    checkboxvalue_G5_12 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G5_12).place(x = 1690+200-25, y = 425, width=30, height=30)
    checkboxvalue_G5_13 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G5_13).place(x = 1690+200-25, y = 460, width=30, height=30)
    checkboxvalue_G5_14 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G5_14).place(x = 1690+200-25, y = 495, width=30, height=30)
    checkboxvalue_G5_15 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G5_15).place(x = 1690+200-25, y = 530, width=30, height=30)
    checkboxvalue_G5_16 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G5_16).place(x = 1690+200-25, y = 565, width=30, height=30)
    checkboxvalue_G5_17 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G5_17).place(x = 1690+200-25, y = 600, width=30, height=30)
    checkboxvalue_G5_18 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G5_18).place(x = 1690+200-25, y = 635, width=30, height=30)
    checkboxvalue_G5_19 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G5_19).place(x = 1690+200-25, y = 670, width=30, height=30)
    checkboxvalue_G5_20 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G5_20).place(x = 1690+200-25, y = 705, width=30, height=30)
    checkboxvalue_G5_21 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G5_21).place(x = 1690+200-25, y = 740, width=30, height=30)
    checkboxvalue_G5_22 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G5_22).place(x = 1690+200-25, y = 775, width=30, height=30)
    checkboxvalue_G5_23 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G5_23).place(x = 1690+200-25, y = 810, width=30, height=30)
    checkboxvalue_G5_24 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G5_24).place(x = 1690+200-25, y = 845, width=30, height=30)
    checkboxvalue_G5_25 = IntVar()
    Checkbutton(root, variable=checkboxvalue_G5_25).place(x = 1690+200-25, y = 880, width=30, height=30)

    

    #G1_master_set_entry = Entry()
    #G1_master_set_entry.place(x = 1700-10+30, y = 880+60, width = 30, height = 25)		
    G1_master_set_entry = Entry()
    G1_master_set_entry.place(x = 1700-10+30, y = 880+80, width = 30, height = 25)		
    G2_master_set_entry = Entry()
    G2_master_set_entry.place(x = 1700-10+30, y = 880+100, width = 30, height = 25)		
    G3_master_set_entry = Entry()
    G3_master_set_entry.place(x = 1700-10+30, y = 880+120, width = 30, height = 25)		
    G4_master_set_entry = Entry()
    G4_master_set_entry.place(x = 1700-10+30, y = 880+140, width = 30, height = 25)		
																																																																																																																																																																																																																																																																																																																																																																																																										
    #.....................group select..................
    checkboxvalue_Group_all = IntVar()
    Checkbutton(root, variable=checkboxvalue_Group_all).place(x = 1700-10, y = 880+60, width=20, height=20)
    checkboxvalue_Group_1 = IntVar()
    Checkbutton(root, variable=checkboxvalue_Group_1).place(x = 1700-10, y = 880+80, width=20, height=20)
    checkboxvalue_Group_2 = IntVar()
    Checkbutton(root, variable=checkboxvalue_Group_2).place(x = 1700-10, y = 880+100, width=20, height=20)
    checkboxvalue_Group_3 = IntVar()
    Checkbutton(root, variable=checkboxvalue_Group_3).place(x = 1700-10, y = 880+120, width=20, height=20)
    checkboxvalue_Group_4 = IntVar()
    Checkbutton(root, variable=checkboxvalue_Group_4).place(x = 1700-10, y = 880+140, width=20, height=20)
    """
    #show_frame()
    # Start the GUI 
    root.mainloop() 

