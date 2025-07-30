"""
Given two dates and the coordinates, download N Sentinel Collections scenes from ESA Sentinel dataHUB.
---------------------------------------------------------------------

Author: Daniel García Díaz & María Peña Fernández
Email: garciad@ifca.unican.es & penam@ifca.unican.es
Institute of Physics of Cantabria (IFCA)
Advanced Computing and e-Science
Date: Sep 2018 - June 2023
"""
#imports apis
import datetime
import xmltodict
import requests
import os
# import pandas as pd
from tqdm import tqdm

# Subfunctions
from .. import config
from ..utils import sat_utils


class download:

    def __init__(self, inidate, enddate, producttype, region_name, platform, output_path,
                 cloud = 0, lim_downloads = None, coordinates = None, footprint = None):
        """
        Parameters
        ----------
        inidate : Initial date of the query in format: datetime.strptime "%Y-%m-%dT%H:%M:%SZ"
        enddate : Final date of the query in format: datetime.strptime "%Y-%m-%dT%H:%M:%SZ"
        coordinates : dict. Coordinates that delimit the region to be searched.
            Example: {"W": -2.830, "S": 41.820, "E": -2.690, "N": 41.910}}
        producttype : str
            Dataset type. A list of productypes can be found in https://mapbox.github.io/usgs/reference/catalog/ee.html
        region_name: str
            Name to save the file
        platform: str
            Name of the mission
        lim_downloads: int
            Number of products wanted to be downloaded
        footprint: geom
            It contains the coordinates of the corners of the poligon containing the AOI
        
        Attention please!!
        ------------------
        Registration and login credentials are required to access all system features and download data.
        To register, please create a username and password.
        Once registered, the username and password must be added to the credentials.yaml file.
        Example: sentinel: {password: password, user: username}
        """

        self.session = requests.Session()

        #Search parameters
        self.inidate = inidate
        self.enddate = enddate
        self.platform = platform.upper() # platform = 'Sentinel-3'
        self.region_name = region_name
        self.producttype = producttype # 'OL_1_EFR___'
        self.coord = coordinates
        self.lim_downloads = lim_downloads
        self.footprint = footprint
        self.cloud = int(cloud)
        self.output_path = output_path

        #work path
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)    
            
        self.output_path = os.path.join(output_path, region_name)
        
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)
            
        #ESA APIs
        self.api_url = 'https://catalogue.dataspace.copernicus.eu/odata/v1/Products?'#'https://scihub.copernicus.eu/dhus/'
        self.credentials = config.load_credentials()['sentinel']

    def get_keycloak(self):
        data = {
            "client_id": "cdse-public",
            "username": self.credentials['user'],
            "password": self.credentials['password'],
            "grant_type": "password",
            }
        try:
            r = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
            data=data,
            )
            r.raise_for_status()
        except Exception as e:
            raise Exception(
                f"Keycloak token creation failed. Reponse from the server was: {r.json()}"
                )
        return r.json()["access_token"]


    
    def search(self, omit_corners=True):
        if self.footprint == None:
            fp = 'POLYGON(({0} {1},{2} {1},{2} {3},{0} {3},{0} {1}))'.format(self.coord['W'],
                                                                                          self.coord['S'],
                                                                                          self.coord['E'],
                                                                                          self.coord['N'])
        else:
            fp = self.footprint

        if self.platform == 'SENTINEL-2':
            url_query = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq '{self.platform}' and OData.CSC.Intersects(area=geography'SRID=4326;{fp}') and ContentDate/Start gt {self.inidate} and ContentDate/Start lt {self.enddate} and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value lt {self.cloud}) and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{self.producttype}')"
        elif self.platform == 'SENTINEL-3':
            url_query = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq '{self.platform}' and OData.CSC.Intersects(area=geography'SRID=4326;{fp}') and ContentDate/Start gt {self.inidate} and ContentDate/Start lt {self.enddate} and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{self.producttype}')"
            
        response = self.session.get(url_query)
        print("Searching with user: %s" % self.credentials['user'])
        response.raise_for_status()
        
        # Parse the response
        json_feed = response.json()#['feed']
        print(url_query)


        if 'value' in json_feed.keys():
            results = json_feed['value']
            if isinstance(results, dict):  # if the query returns only one product, products will be a dict not a list
                results = [results]
        else:
            results = []

        # Remove results that are mainly corners
        def keep(r):
            if r['ContentLength'] > 0.5e9: #500MB
                return True
            else:
                return False
                
        results[:] = [r for r in results if keep(r)]
        print('Retrieving {} results \n'.format(len(results)))
        return results

    def download(self):
        
        #results of the search
        results = self.search()

        print('Trying to download {} out of {} results \n'.format(self.lim_downloads, len(results)))
        
        if self.lim_downloads != None and len(results) > self.lim_downloads:
            results = results[:self.lim_downloads]

        downloaded, pending= [], []
        keycloak_token = self.get_keycloak()
        session = requests.Session()
        session.headers.update({'Authorization': f'Bearer {keycloak_token}'})
        print("Authorized OK")
        
        for result in results:
            print(f"Product online? {result['Online']}")            
            if result['Online'] and (self.producttype):
                
                url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products(%s)/$value" % result['Id']
                tile_id = result['Name']
                if self.platform == 'SENTINEL-2':
                    wrs = (tile_id.split('_'))[5]
                elif self.platform == 'SENTINEL-3':
                    wrs = (tile_id.split('_'))[13]
                date = sat_utils.get_date(tile_id, satellite = self.platform)
                print ('Tile {} ... date {} ... wrs {}'.format(tile_id, date, wrs))
                
                if  os.path.isdir('%s/%s.SEN3' % (self.output_path, tile_id)):
                    print ('Already downloaded \n')
                    break
                    
                else:
                    print(f"Downloading {url}")
                    response = session.head(url, allow_redirects=False)                    
                    
                    if response.status_code in (301, 302, 303, 307):
                        url = response.headers['Location']
                        print(url)
                        #response = session.get(url, allow_redirects=False)
                    response = session.get(url, stream=True)

                    print(f"Status code {response.status_code}")
                    
                    if response.status_code == 200:
    
                        total_size = int(response.headers.get('content-length', 0))
                        chunk_size = 1024  # Define the chunk size
    
                        # Initialize an empty byte array to hold the data
                        data = bytearray()
    
                        with tqdm(
                            desc="Downloading",
                            total=total_size,
                            unit='iB',
                            unit_scale=True,
                            unit_divisor=1024,
                        ) as bar:
                            for chunk in response.iter_content(chunk_size=chunk_size):
                                # Update the byte array with the chunk
                                data.extend(chunk)
                                # Update the progress bar
                                bar.update(len(chunk))

                            downloaded.append('%s/%s' % (self.output_path, tile_id))
                        # downloaded.append(result['Name'])
    
                        print('Downloading {} ... \n'.format(result['Name']))
    
                        sat_utils.open_compressed(byte_stream=data,
                                                  file_format='zip',
                                                  output_folder=self.output_path)
                        
                        # os.rename('%s\%s.SEN3' % (self.output_path, tile_id), 
                        #   '%s\%s.SEN3' % (self.output_path, self.region_name))
  
                        print('Saved as... %s/%s' % (self.output_path, tile_id))
                
                    else:
                        pending[self.region_name] = {'tile_id': tile_id, 'url': url}
                        print ('The product is offline')
                        print ('Activating recovery mode ...')
                    
            else:
                pending[self.region_name] = {'tile_id': tile_id, 'url': url}
                print ('The product {} is offline'.format(result['Name']))
                print ('Activating recovery mode ... \n')
                print ('{} \n'.format(url))
                
        return downloaded, pending
