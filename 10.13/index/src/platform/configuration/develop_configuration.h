/*
 * Forwarding header: some `engine/` sources include "configuration/..." while
 * others include "src/configuration/...". Both resolve to the same stub.
 */
#pragma once
#include "src/configuration/develop_configuration.h"
