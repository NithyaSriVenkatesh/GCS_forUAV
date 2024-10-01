# Import necessary packages
import math
import numpy as np
import pandas as pd
import gridNodes, triangularNodes, lineNodes
import locatePosition as locate
from scipy import interpolate

def grid(numDrones, distNodes, origin):
    # This function differs for other formation
    nodes = gridNodes.nodesGeneration(numDrones, distNodes)

    incrementalDistance = 1000

    # Converting nodes (numpy array) to pandas array for the sake of calculation
    df = pd.DataFrame(data = nodes, columns = ['X','Y'])

    # The initial point of rectangle in (x,y) is (0,0) so considering the current
    # location as origin and retreiving the latitude and longitude from the GPS
    # origin = (12.948048, 80.139742)

    # Calculating the hypot end point for interpolating the latitudes and longitudes 
    rEndDistance = math.sqrt(2*(incrementalDistance**2))

    # The bearing for the hypot angle is 45 degrees considering coverage area as square
    bearing = 45

    # Determining the Latitude and Longitude of Middle point of the sqaure area
    #  and hypot end point of square area for interpolating latitude and longitude
    rMiddle, rEnd = locate.destination_location(origin[0], origin[1], rEndDistance/2, bearing), locate.destination_location(origin[0], origin[1], rEndDistance, bearing)

    # Array of (x,y)
    x_cart, y_cart  = [0, incrementalDistance/2, incrementalDistance], [0, incrementalDistance/2, incrementalDistance]

    # Array of (latitude, longitude)
    x_lon, y_lat = [origin[1], rMiddle[1], rEnd[1]], [origin[0], rMiddle[0], rEnd[0]]

    # Latitude interpolation function 
    f_lat = interpolate.interp1d(y_cart, y_lat)

    # Longitude interpolation function
    f_lon = interpolate.interp1d(x_cart, x_lon)

    # Splitting the columns of dataframe (nodes) as x and y
    x, y = df.loc[:,'X'], df.loc[:,'Y']

    # Converting (x,y) to (latitude, longitude) using interpolation function
    lat, lon = f_lat(y), f_lon(x)

    # Arranging the coordinates in (latitude, longitude) format into a single array
    nodeLocation = np.vstack([lat.ravel(), lon.ravel()]).T

    return nodeLocation


def triangle(numDrones, distNodes, origin):
    nodes = triangularNodes.nodesGeneration(numDrones, distNodes)

    coverageDistance = 1000

    # Converting nodes (numpy array) to pandas array for the sake of calculation
    df = pd.DataFrame(data = nodes, columns = ['X','Y'])

    # The initial point of rectangle in (x,y) is (0,0) so considering the current
    # location as origin and retreiving the latitude and longitude from the GPS
    # origin = (12.948048, 80.139742)

    # Calculating the hypot end point for interpolating the latitudes and longitudes 
    rEndDistance = math.sqrt(2*(coverageDistance**2))

    # The bearing for the hypot angle is 45 degrees considering coverage area as square
    bearing = 45

    # Determining the Latitude and Longitude of Middle point of the sqaure area
    # and hypot end point of square area for interpolating latitude and longitude
    rMiddle, rEnd = locate.destination_location(origin[0], origin[1], rEndDistance/2, bearing), locate.destination_location(origin[0], origin[1], rEndDistance, bearing)

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

    # Converting (x,y) to (latitude, longitude) using interpolation function
    lat, lon = f_lat(y), f_lon(x)

    # plt.plot(lat, lon,'-o', color='black')
    # plt.axis('equal')

    # Arranging the coordinates in (latitude, longitude) format into a single array
    nodeLocation = np.vstack([lat.ravel(), lon.ravel()]).T

    return nodeLocation

def line(numDrones, distNodes, origin, lineType):
    nodes, flag = lineNodes.nodesGeneration(numDrones, distNodes, lineType)
    coverageDistance = 1000
    if flag != False:
        # Converting nodes (numpy array) to pandas array for the sake of calculation
        df = pd.DataFrame(data = nodes, columns = ['X','Y'])

        # Retreiving the latitude and longitude from the GPS
        #origin = (12.9480474, 80.1397414)
        
        # The bearing for the horizontal line is 90 and vertical line is 0
        if lineType == 'H':
            bearing = 90

        if lineType == 'V':
            bearing = 0

        # Determining the Latitude and Longitude of Middle point of the sqaure area
        # and hypot end point of square area for interpolating latitude and longitude
        rMiddle, rEnd = locate.destination_location(origin[0], origin[1], coverageDistance/2, bearing), locate.destination_location(origin[0], origin[1], coverageDistance, bearing)

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

        # Converting (x,y) to (latitude, longitude) using interpolation function
        lat, lon = f_lat(y), f_lon(x)

        # Arranging the coordinates in (latitude, longitude) format into a single array
        nodeLocation = np.vstack([lat.ravel(), lon.ravel()]).T

        return nodeLocation