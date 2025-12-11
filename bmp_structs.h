#ifndef BMP_STRUCTS_H
#define BMP_STRUCTS_H

#include <stdint.h>

typedef struct {
    uint8_t b, g, r; // Note: BMP BGR
} pixel;

#pragma pack(push, 1) // Tell the compiler not to change the alignment or size of the struct

// Header default for BMP 24-bit
typedef struct {
    uint16_t type;              
    uint32_t size;              
    uint16_t reserved1;         
    uint16_t reserved2;         
    uint32_t offset;            
    uint32_t header_size;       
    int32_t  width;             
    int32_t  height;            
    uint16_t planes;            
    uint16_t bits;              
    uint32_t compression;       
    uint32_t imagesize;         
    int32_t  xresolution;       
    int32_t  yresolution;       
    uint32_t ncolours;          
    uint32_t importantcolours;  
} bmpHeader;

#pragma pack(pop) // Restores compiler behavior

#endif