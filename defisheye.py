#!/usr/bin/env python3.6
# -*- Coding: UTF-8 -*-
"""
Defisheye algorithm.

Developed by: E. S. Pereira.
e-mail: pereira.somoza@gmail.com

Based in the work of F. Weinhaus.
http://www.fmwconcepts.com/imagemagick/defisheye/index.php

Copyright [2019] [E. S. Pereira]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import cv2
from numpy import arange, sqrt, arctan, sin, tan, zeros, array, meshgrid, pi, uint8, arcsin
from numpy import argwhere, hypot


class Defisheye:
    """
    Defisheye

    fov: fisheye field of view (aperture) in degrees. ffoc(fusheye focal length) is computed from this value.
    p_radius: radius of perspective area. defish된 이미지의 반지름.
    pfov: perspective field of view (aperture) in degrees. defish해서 보여질 영역의 화각.
    xcenter: x center of fisheye area
    ycenter: y center of fisheye area
    radius: radius of fisheye area
    angle: image rotation in degrees clockwise
    dtype: linear, equalarea, orthographic, stereographic
    format: circular, fullframe
    """

    def __init__(self, infile, **kwargs):
        vkwargs = {"fov": 180,
                   "pfov": 120,
                   "xcenter": None,
                   "ycenter": None,
                   "radius": None,
                   "p_radius": None,
                   "angle": 0,
                   "dtype": "equalarea",
                   "format": "circular"
                   }
        self._start_att(vkwargs, kwargs)

        if type(infile) == str:
            _image = cv2.imread(infile)
        elif type(infile) == ndarray:
            _image = infile
        else:
            raise ImageError("Image format not recognized")

        self._image = _image
        width = _image.shape[1]
        height = _image.shape[0]

        if self._xcenter is None:
            self._xcenter = (width - 1) // 2

        if self._ycenter is None:
            self._ycenter = (height - 1) // 2

        if self._radius is None:
            self._radius = (min(width, height) - 1) // 2

        self._width = self._image.shape[1]
        self._height = self._image.shape[0]

        print(self._xcenter, self._ycenter)

        print(self._width, self._height)
        print(self._radius, self._p_radius)

        if self._dtype == "linear":
            self._ffoc = self._radius * 2 * 180 / (self._fov * pi)
        elif self._dtype == "equalarea":
            self._ffoc = self._radius * 2 / (2.0 * sin(self._fov * pi / 720))
        elif self._dtype == "orthographic":
            self._ffoc = self._radius * 2 / (2.0 * sin(self._fov * pi / 360))
        elif self._dtype == "stereographic":
            self._ffoc = self._radius * 2 / (2.0 * tan(self._fov * pi / 720))

    def _map(self, i, j, pfocinv):

        xd = i - self._p_radius
        yd = j - self._p_radius

        rd = hypot(xd, yd)
        phiang = arctan(pfocinv * rd)

        if self._dtype == "linear":
            rr = self._ffoc * phiang
            # rr = "rr={}*phiang;".format(ifoc)

        elif self._dtype == "equalarea":
            rr = self._ffoc * sin(phiang / 2)
            # rr = "rr={}*sin(phiang/2);".format(ifoc)

        elif self._dtype == "orthographic":
            rr = self._ffoc * sin(phiang)
            # rr="rr={}*sin(phiang);".format(ifoc)

        elif self._dtype == "stereographic":
            rr = self._ffoc * tan(phiang / 2)
        #rr = rr * 4 / 3
        rdmask = rd != 0
        xs = xd.copy()
        ys = yd.copy()

        xs[rdmask] = (rr[rdmask] / rd[rdmask]) * xd[rdmask] + self._xcenter
        ys[rdmask] = (rr[rdmask] / rd[rdmask]) * yd[rdmask] + self._ycenter

        xs[~rdmask] = 0
        ys[~rdmask] = 0

        xs = xs.astype(int)
        ys = ys.astype(int)
        
        return xs, ys

    def convert(self, outfile):

        # compute output (perspective) focal length and its inverse from ofov
        # phi=fov/2; r=N/2
        # r/f=tan(phi);
        # f=r/tan(phi);
        # f= (N/2)/tan((fov/2)*(pi/180)) = N/(2*tan(fov*pi/360))
        dim = self._p_radius * 2
        self._pfoc = dim / (2 * tan(self._pfov * pi / 360))
        pfocinv = 1.0 / self._pfoc

        i = arange(dim)
        j = arange(dim)
        i, j = meshgrid(i, j)

        print(i)
        print(j)
        xs, ys, = self._map(i, j, pfocinv)
                
        img = zeros((dim, dim, 3), dtype=uint8)
        img[j, i] = self._image[ys, xs]
        cv2.imwrite(outfile, img)
        return img

    def _start_att(self, vkwargs, kwargs):
        """
        Starting atributes
        """
        pin = []

        for key, value in kwargs.items():
            if key not in vkwargs:
                raise NameError("Invalid key {}".format(key))
            else:
                pin.append(key)
                setattr(self, "_{}".format(key), value)

        pin = set(pin)
        rkeys = set(vkwargs.keys()) - pin
        for key in rkeys:
            setattr(self, "_{}".format(key), vkwargs[key])

    def convert_point(self, fx, fy):
        """
        convert fisheye view point to perspective view point
        """
        x0 = fx - self._xcenter
        y0 = fy - self._ycenter
        r0 = hypot(x0, y0)
        phiang = 2 * arcsin(r0 * 3 / 4/ self._ffoc)
        r = self._pfoc * tan(phiang)
        x = x0 * r / r0 + self._p_radius
        y = y0 * r / r0 + self._p_radius
        return int(x), int(y)


