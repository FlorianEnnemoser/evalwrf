import evalwrf as ew

def main():
    wrfp = "D:\\Python\\evalwrf\\wrfout_d03_2024-06-09_00_00_00"
    wrfi = "D:\\Python\\evalwrf\\wrfinput_d03"
    ds = ew.pre.wrf_to_xr(wrfp,wrfi)
    print(ds)





if __name__ == "__main__":
    main()
