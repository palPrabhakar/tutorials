/*
 * A simple KMS example following:
 * https://www.youtube.com/watch?v=haes4_Xnc5Q
 */
#include <fcntl.h>
#include <libdrm/drm_fourcc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#define DEVICE "/dev/dri/card1"

static uint64_t get_property_value(int drm_fd, uint32_t object_id,
                                   uint32_t object_type,
                                   const char *prop_name) {
    drmModeObjectProperties *props =
        drmModeObjectGetProperties(drm_fd, object_id, object_type);
    for (uint32_t i = 0; i < props->count_props; i++) {
        drmModePropertyRes *prop = drmModeGetProperty(drm_fd, props->props[i]);
        uint64_t val = props->prop_values[i];
        if (strcmp(prop->name, prop_name) == 0) {
            drmModeFreeProperty(prop);
            drmModeFreeObjectProperties(props);
            return val;
        }
        drmModeFreeProperty(prop);
    }
    fprintf(stderr, "Unable to find property %s\n", prop_name);
    exit(1);
}

void add_property(int drm_fd, drmModeAtomicReq *req, uint32_t object_id,
                  uint32_t object_type, const char *prop_name, uint64_t value) {
    drmModeObjectProperties *props =
        drmModeObjectGetProperties(drm_fd, object_id, object_type);
    for (uint32_t i = 0; i < props->count_props; i++) {
        drmModePropertyRes *prop = drmModeGetProperty(drm_fd, props->props[i]);
        if (strcmp(prop->name, prop_name) == 0) {
            drmModeAtomicAddProperty(req, object_id, prop->prop_id, value);
            drmModeFreeProperty(prop);
            return;
        }
        drmModeFreeProperty(prop);
    }
    fprintf(stderr, "Unable to find property %s\n", prop_name);
    exit(1);
}

int main(void) {
    printf("DRM-KMS\n");

    int drm_fd = open(DEVICE, O_RDWR | O_NONBLOCK);
    if (drm_fd == -1) {
        perror("Open failed");
        return 1;
    }

    // set client capabilities
    if (drmSetClientCap(drm_fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1)) {
        perror("drmSetClientCap(UNIVERSAL_PLANES) failed");
        return 1;
    }
    if (drmSetClientCap(drm_fd, DRM_CLIENT_CAP_ATOMIC, 1)) {
        perror("drmSetClientCap(ATOMIC) failed");
        return 1;
    }

    drmModeRes *resources = drmModeGetResources(drm_fd);

    // Get the first CRTC currently lighted up
    drmModeCrtc *crtc = NULL;
    for (int i = 0; i < resources->count_crtcs; ++i) {
        crtc = drmModeGetCrtc(drm_fd, resources->crtcs[i]);
        if (crtc->mode_valid)
            break;
        drmModeFreeCrtc(crtc);
        crtc = NULL;
    }
    if (!crtc) {
        fprintf(stderr, "Unable to find CRTC\n");
        return 1;
    }
    printf("Using CRTC %u\n", crtc->crtc_id);
    printf("Using mode %dx%d %dHz\n", crtc->mode.hdisplay, crtc->mode.vdisplay,
           crtc->mode.vrefresh);

    // Get the primary plane connected to the CRTC
    drmModePlaneRes *planes = drmModeGetPlaneResources(drm_fd);
    drmModePlane *plane = NULL;
    for (uint32_t i = 0; i < planes->count_planes; i++) {
        plane = drmModeGetPlane(drm_fd, planes->planes[i]);
        uint64_t plane_type = get_property_value(drm_fd, planes->planes[i],
                                                 DRM_MODE_OBJECT_PLANE, "type");
        if (plane->crtc_id == crtc->crtc_id &&
            plane_type == DRM_PLANE_TYPE_PRIMARY) {
            break;
        }
        drmModeFreePlane(plane);
        plane = NULL;
    }
    if (!plane) {
        fprintf(stderr, "Unable to find a primary plane\n");
        return 1;
    }
    printf("Using plane %u\n", plane->plane_id);

    drmModeFreePlaneResources(planes);
    drmModeFreeResources(resources);

    // Allocate a buffer and get a driver-specific handle back
    int width = crtc->mode.hdisplay;
    int height = crtc->mode.vdisplay;
    struct drm_mode_create_dumb create = {
        .width = width,
        .height = height,
        .bpp = 32,
    };
    drmIoctl(drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &create);

    // Create the DRM framebuffer object
    uint32_t handles[4] = {create.handle};
    uint32_t strides[4] = {create.pitch};
    uint32_t offsets[4] = {0};
    uint32_t fb_id = 0;
    drmModeAddFB2(drm_fd, width, height, DRM_FORMAT_XRGB8888, handles, strides,
                  offsets, &fb_id, 0);
    printf("Allocated FB %u\n", fb_id);

    // Create a memory mapping
    struct drm_mode_map_dumb map = {.handle = create.handle};
    drmIoctl(drm_fd, DRM_IOCTL_MODE_MAP_DUMB, &map);
    uint8_t *data = mmap(0, create.size, PROT_READ | PROT_WRITE, MAP_SHARED,
                         drm_fd, map.offset);

    uint8_t color[] = {0xAE, 0xE7, 0xFC, 0xFF}; // B, G, R, X
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t offset = y * create.pitch + x * sizeof(color);
            memcpy(&data[offset], color, sizeof(color));
        }
    }

    // Submit an atomic commit
    drmModeAtomicReq *req = drmModeAtomicAlloc();

    add_property(drm_fd, req, plane->plane_id, DRM_MODE_OBJECT_PLANE, "FB_ID",
                 fb_id);
    add_property(drm_fd, req, plane->plane_id, DRM_MODE_OBJECT_PLANE, "SRC_X",
                 0);
    add_property(drm_fd, req, plane->plane_id, DRM_MODE_OBJECT_PLANE, "SRC_Y",
                 0);
    add_property(drm_fd, req, plane->plane_id, DRM_MODE_OBJECT_PLANE, "SRC_W",
                 width << 16);
    add_property(drm_fd, req, plane->plane_id, DRM_MODE_OBJECT_PLANE, "SRC_H",
                 height << 16);
    add_property(drm_fd, req, plane->plane_id, DRM_MODE_OBJECT_PLANE, "CRTC_X",
                 0);
    add_property(drm_fd, req, plane->plane_id, DRM_MODE_OBJECT_PLANE, "CRTC_Y",
                 0);
    add_property(drm_fd, req, plane->plane_id, DRM_MODE_OBJECT_PLANE, "CRTC_W",
                 width);
    add_property(drm_fd, req, plane->plane_id, DRM_MODE_OBJECT_PLANE, "CRTC_H",
                 height);

    uint32_t flags = DRM_MODE_ATOMIC_NONBLOCK;
    int ret = drmModeAtomicCommit(drm_fd, req, flags, NULL);
    if (ret != 0) {
        perror("drmModeAtomicCommit failed");
        return 1;
    }

    // Sleep for a while so that we can see the result on screen
    sleep(5);

    return 0;
}
