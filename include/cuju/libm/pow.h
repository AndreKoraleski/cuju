// Copyright (c) 2017-2020, Arm Limited.
//
// Ported to C++ for host and device from Arm Optimized Routines v21.02 math/pow.c,
// pow_log_data.c, exp_data.c, math_err.c and math_config.h by the cuju Authors, 2026. glibc 2.39
// carries the same code in sysdeps/ieee754/dbl-64/e_pow.c.
// SPDX-License-Identifier: MIT

// Double pow of glibc 2.39's build for FMA and AVX2, bit for bit on host and device.
// Exact: each `fma` is an explicit fused step or a contraction of that build; nothing else fuses.
#pragma once

#include "../host_device.h"
#include "common.h"

namespace cuju {
namespace libm {

// `logarithmOfCenter` and its tail carry log(c) to more than double precision.
struct PowLogarithmEntry {
    double inverseCenter, logarithmOfCenter, logarithmOfCenterTail;
};

// `__pow_log_data.tab` without its padding: 1 / c and log(c) for 128 centers c around 1.
CUJU_TABLE_DEFINITION(PowLogarithmEntry, POW_LOGARITHM_TABLE,
                      {
                          {0x1.6a00000000000p+0, -0x1.62c82f2b9c800p-2, 0x1.ab42428375680p-48},
                          {0x1.6800000000000p+0, -0x1.5d1bdbf580800p-2, -0x1.ca508d8e0f720p-46},
                          {0x1.6600000000000p+0, -0x1.5767717455800p-2, -0x1.362a4d5b6506dp-45},
                          {0x1.6400000000000p+0, -0x1.51aad872df800p-2, -0x1.684e49eb067d5p-49},
                          {0x1.6200000000000p+0, -0x1.4be5f95777800p-2, -0x1.41b6993293ee0p-47},
                          {0x1.6000000000000p+0, -0x1.4618bc21c6000p-2, 0x1.3d82f484c84ccp-46},
                          {0x1.5e00000000000p+0, -0x1.404308686a800p-2, 0x1.c42f3ed820b3ap-50},
                          {0x1.5c00000000000p+0, -0x1.3a64c55694800p-2, 0x1.0b1c686519460p-45},
                          {0x1.5a00000000000p+0, -0x1.347dd9a988000p-2, 0x1.5594dd4c58092p-45},
                          {0x1.5800000000000p+0, -0x1.2e8e2bae12000p-2, 0x1.67b1e99b72bd8p-45},
                          {0x1.5600000000000p+0, -0x1.2895a13de8800p-2, 0x1.5ca14b6cfb03fp-46},
                          {0x1.5600000000000p+0, -0x1.2895a13de8800p-2, 0x1.5ca14b6cfb03fp-46},
                          {0x1.5400000000000p+0, -0x1.22941fbcf7800p-2, -0x1.65a242853da76p-46},
                          {0x1.5200000000000p+0, -0x1.1c898c1699800p-2, -0x1.fafbc68e75404p-46},
                          {0x1.5000000000000p+0, -0x1.1675cababa800p-2, 0x1.f1fc63382a8f0p-46},
                          {0x1.4e00000000000p+0, -0x1.1058bf9ae4800p-2, -0x1.6a8c4fd055a66p-45},
                          {0x1.4c00000000000p+0, -0x1.0a324e2739000p-2, -0x1.c6bee7ef4030ep-47},
                          {0x1.4a00000000000p+0, -0x1.0402594b4d000p-2, -0x1.036b89ef42d7fp-48},
                          {0x1.4a00000000000p+0, -0x1.0402594b4d000p-2, -0x1.036b89ef42d7fp-48},
                          {0x1.4800000000000p+0, -0x1.fb9186d5e4000p-3, 0x1.d572aab993c87p-47},
                          {0x1.4600000000000p+0, -0x1.ef0adcbdc6000p-3, 0x1.b26b79c86af24p-45},
                          {0x1.4400000000000p+0, -0x1.e27076e2af000p-3, -0x1.72f4f543fff10p-46},
                          {0x1.4200000000000p+0, -0x1.d5c216b4fc000p-3, 0x1.1ba91bbca681bp-45},
                          {0x1.4000000000000p+0, -0x1.c8ff7c79aa000p-3, 0x1.7794f689f8434p-45},
                          {0x1.4000000000000p+0, -0x1.c8ff7c79aa000p-3, 0x1.7794f689f8434p-45},
                          {0x1.3e00000000000p+0, -0x1.bc286742d9000p-3, 0x1.94eb0318bb78fp-46},
                          {0x1.3c00000000000p+0, -0x1.af3c94e80c000p-3, 0x1.a4e633fcd9066p-52},
                          {0x1.3a00000000000p+0, -0x1.a23bc1fe2b000p-3, -0x1.58c64dc46c1eap-45},
                          {0x1.3a00000000000p+0, -0x1.a23bc1fe2b000p-3, -0x1.58c64dc46c1eap-45},
                          {0x1.3800000000000p+0, -0x1.9525a9cf45000p-3, -0x1.ad1d904c1d4e3p-45},
                          {0x1.3600000000000p+0, -0x1.87fa06520d000p-3, 0x1.bbdbf7fdbfa09p-45},
                          {0x1.3400000000000p+0, -0x1.7ab890210e000p-3, 0x1.bdb9072534a58p-45},
                          {0x1.3400000000000p+0, -0x1.7ab890210e000p-3, 0x1.bdb9072534a58p-45},
                          {0x1.3200000000000p+0, -0x1.6d60fe719d000p-3, -0x1.0e46aa3b2e266p-46},
                          {0x1.3000000000000p+0, -0x1.5ff3070a79000p-3, -0x1.e9e439f105039p-46},
                          {0x1.3000000000000p+0, -0x1.5ff3070a79000p-3, -0x1.e9e439f105039p-46},
                          {0x1.2e00000000000p+0, -0x1.526e5e3a1b000p-3, -0x1.0de8b90075b8fp-45},
                          {0x1.2c00000000000p+0, -0x1.44d2b6ccb8000p-3, 0x1.70cc16135783cp-46},
                          {0x1.2c00000000000p+0, -0x1.44d2b6ccb8000p-3, 0x1.70cc16135783cp-46},
                          {0x1.2a00000000000p+0, -0x1.371fc201e9000p-3, 0x1.178864d27543ap-48},
                          {0x1.2800000000000p+0, -0x1.29552f81ff000p-3, -0x1.48d301771c408p-45},
                          {0x1.2600000000000p+0, -0x1.1b72ad52f6000p-3, -0x1.e80a41811a396p-45},
                          {0x1.2600000000000p+0, -0x1.1b72ad52f6000p-3, -0x1.e80a41811a396p-45},
                          {0x1.2400000000000p+0, -0x1.0d77e7cd09000p-3, 0x1.a699688e85bf4p-47},
                          {0x1.2400000000000p+0, -0x1.0d77e7cd09000p-3, 0x1.a699688e85bf4p-47},
                          {0x1.2200000000000p+0, -0x1.fec9131dbe000p-4, -0x1.575545ca333f2p-45},
                          {0x1.2000000000000p+0, -0x1.e27076e2b0000p-4, 0x1.a342c2af0003cp-45},
                          {0x1.2000000000000p+0, -0x1.e27076e2b0000p-4, 0x1.a342c2af0003cp-45},
                          {0x1.1e00000000000p+0, -0x1.c5e548f5bc000p-4, -0x1.d0c57585fbe06p-46},
                          {0x1.1c00000000000p+0, -0x1.a926d3a4ae000p-4, 0x1.53935e85baac8p-45},
                          {0x1.1c00000000000p+0, -0x1.a926d3a4ae000p-4, 0x1.53935e85baac8p-45},
                          {0x1.1a00000000000p+0, -0x1.8c345d631a000p-4, 0x1.37c294d2f5668p-46},
                          {0x1.1a00000000000p+0, -0x1.8c345d631a000p-4, 0x1.37c294d2f5668p-46},
                          {0x1.1800000000000p+0, -0x1.6f0d28ae56000p-4, -0x1.69737c93373dap-45},
                          {0x1.1600000000000p+0, -0x1.51b073f062000p-4, 0x1.f025b61c65e57p-46},
                          {0x1.1600000000000p+0, -0x1.51b073f062000p-4, 0x1.f025b61c65e57p-46},
                          {0x1.1400000000000p+0, -0x1.341d7961be000p-4, 0x1.c5edaccf913dfp-45},
                          {0x1.1400000000000p+0, -0x1.341d7961be000p-4, 0x1.c5edaccf913dfp-45},
                          {0x1.1200000000000p+0, -0x1.16536eea38000p-4, 0x1.47c5e768fa309p-46},
                          {0x1.1000000000000p+0, -0x1.f0a30c0118000p-5, 0x1.d599e83368e91p-45},
                          {0x1.1000000000000p+0, -0x1.f0a30c0118000p-5, 0x1.d599e83368e91p-45},
                          {0x1.0e00000000000p+0, -0x1.b42dd71198000p-5, 0x1.c827ae5d6704cp-46},
                          {0x1.0e00000000000p+0, -0x1.b42dd71198000p-5, 0x1.c827ae5d6704cp-46},
                          {0x1.0c00000000000p+0, -0x1.77458f632c000p-5, -0x1.cfc4634f2a1eep-45},
                          {0x1.0c00000000000p+0, -0x1.77458f632c000p-5, -0x1.cfc4634f2a1eep-45},
                          {0x1.0a00000000000p+0, -0x1.39e87b9fec000p-5, 0x1.502b7f526feaap-48},
                          {0x1.0a00000000000p+0, -0x1.39e87b9fec000p-5, 0x1.502b7f526feaap-48},
                          {0x1.0800000000000p+0, -0x1.f829b0e780000p-6, -0x1.980267c7e09e4p-45},
                          {0x1.0800000000000p+0, -0x1.f829b0e780000p-6, -0x1.980267c7e09e4p-45},
                          {0x1.0600000000000p+0, -0x1.7b91b07d58000p-6, -0x1.88d5493faa639p-45},
                          {0x1.0400000000000p+0, -0x1.fc0a8b0fc0000p-7, -0x1.f1e7cf6d3a69cp-50},
                          {0x1.0400000000000p+0, -0x1.fc0a8b0fc0000p-7, -0x1.f1e7cf6d3a69cp-50},
                          {0x1.0200000000000p+0, -0x1.fe02a6b100000p-8, -0x1.9e23f0dda40e4p-46},
                          {0x1.0200000000000p+0, -0x1.fe02a6b100000p-8, -0x1.9e23f0dda40e4p-46},
                          {0x1.0000000000000p+0, 0x0.0000000000000p+0, 0x0.0000000000000p+0},
                          {0x1.0000000000000p+0, 0x0.0000000000000p+0, 0x0.0000000000000p+0},
                          {0x1.fc00000000000p-1, 0x1.0101575890000p-7, -0x1.0c76b999d2be8p-46},
                          {0x1.f800000000000p-1, 0x1.0205658938000p-6, -0x1.3dc5b06e2f7d2p-45},
                          {0x1.f400000000000p-1, 0x1.8492528c90000p-6, -0x1.aa0ba325a0c34p-45},
                          {0x1.f000000000000p-1, 0x1.0415d89e74000p-5, 0x1.111c05cf1d753p-47},
                          {0x1.ec00000000000p-1, 0x1.466aed42e0000p-5, -0x1.c167375bdfd28p-45},
                          {0x1.e800000000000p-1, 0x1.894aa149fc000p-5, -0x1.97995d05a267dp-46},
                          {0x1.e400000000000p-1, 0x1.ccb73cdddc000p-5, -0x1.a68f247d82807p-46},
                          {0x1.e200000000000p-1, 0x1.eea31c006c000p-5, -0x1.e113e4fc93b7bp-47},
                          {0x1.de00000000000p-1, 0x1.1973bd1466000p-4, -0x1.5325d560d9e9bp-45},
                          {0x1.da00000000000p-1, 0x1.3bdf5a7d1e000p-4, 0x1.cc85ea5db4ed7p-45},
                          {0x1.d600000000000p-1, 0x1.5e95a4d97a000p-4, -0x1.c69063c5d1d1ep-45},
                          {0x1.d400000000000p-1, 0x1.700d30aeac000p-4, 0x1.c1e8da99ded32p-49},
                          {0x1.d000000000000p-1, 0x1.9335e5d594000p-4, 0x1.3115c3abd47dap-45},
                          {0x1.cc00000000000p-1, 0x1.b6ac88dad6000p-4, -0x1.390802bf768e5p-46},
                          {0x1.ca00000000000p-1, 0x1.c885801bc4000p-4, 0x1.646d1c65aacd3p-45},
                          {0x1.c600000000000p-1, 0x1.ec739830a2000p-4, -0x1.dc068afe645e0p-45},
                          {0x1.c400000000000p-1, 0x1.fe89139dbe000p-4, -0x1.534d64fa10afdp-45},
                          {0x1.c000000000000p-1, 0x1.1178e8227e000p-3, 0x1.1ef78ce2d07f2p-45},
                          {0x1.be00000000000p-1, 0x1.1aa2b7e23f000p-3, 0x1.ca78e44389934p-45},
                          {0x1.ba00000000000p-1, 0x1.2d1610c868000p-3, 0x1.39d6ccb81b4a1p-47},
                          {0x1.b800000000000p-1, 0x1.365fcb0159000p-3, 0x1.62fa8234b7289p-51},
                          {0x1.b400000000000p-1, 0x1.4913d8333b000p-3, 0x1.5837954fdb678p-45},
                          {0x1.b200000000000p-1, 0x1.527e5e4a1b000p-3, 0x1.633e8e5697dc7p-45},
                          {0x1.ae00000000000p-1, 0x1.6574ebe8c1000p-3, 0x1.9cf8b2c3c2e78p-46},
                          {0x1.ac00000000000p-1, 0x1.6f0128b757000p-3, -0x1.5118de59c21e1p-45},
                          {0x1.aa00000000000p-1, 0x1.7898d85445000p-3, -0x1.c661070914305p-46},
                          {0x1.a600000000000p-1, 0x1.8beafeb390000p-3, -0x1.73d54aae92cd1p-47},
                          {0x1.a400000000000p-1, 0x1.95a5adcf70000p-3, 0x1.7f22858a0ff6fp-47},
                          {0x1.a000000000000p-1, 0x1.a93ed3c8ae000p-3, -0x1.8724350562169p-45},
                          {0x1.9e00000000000p-1, 0x1.b31d8575bd000p-3, -0x1.c358d4eace1aap-47},
                          {0x1.9c00000000000p-1, 0x1.bd087383be000p-3, -0x1.d4bc4595412b6p-45},
                          {0x1.9a00000000000p-1, 0x1.c6ffbc6f01000p-3, -0x1.1ec72c5962bd2p-48},
                          {0x1.9600000000000p-1, 0x1.db13db0d49000p-3, -0x1.aff2af715b035p-45},
                          {0x1.9400000000000p-1, 0x1.e530effe71000p-3, 0x1.212276041f430p-51},
                          {0x1.9200000000000p-1, 0x1.ef5ade4dd0000p-3, -0x1.a211565bb8e11p-51},
                          {0x1.9000000000000p-1, 0x1.f991c6cb3b000p-3, 0x1.bcbecca0cdf30p-46},
                          {0x1.8c00000000000p-1, 0x1.07138604d5800p-2, 0x1.89cdb16ed4e91p-48},
                          {0x1.8a00000000000p-1, 0x1.0c42d67616000p-2, 0x1.7188b163ceae9p-45},
                          {0x1.8800000000000p-1, 0x1.1178e8227e800p-2, -0x1.c210e63a5f01cp-45},
                          {0x1.8600000000000p-1, 0x1.16b5ccbacf800p-2, 0x1.b9acdf7a51681p-45},
                          {0x1.8400000000000p-1, 0x1.1bf99635a6800p-2, 0x1.ca6ed5147bdb7p-45},
                          {0x1.8200000000000p-1, 0x1.214456d0eb800p-2, 0x1.a87deba46baeap-47},
                          {0x1.7e00000000000p-1, 0x1.2bef07cdc9000p-2, 0x1.a9cfa4a5004f4p-45},
                          {0x1.7c00000000000p-1, 0x1.314f1e1d36000p-2, -0x1.8e27ad3213cb8p-45},
                          {0x1.7a00000000000p-1, 0x1.36b6776be1000p-2, 0x1.16ecdb0f177c8p-46},
                          {0x1.7800000000000p-1, 0x1.3c25277333000p-2, 0x1.83b54b606bd5cp-46},
                          {0x1.7600000000000p-1, 0x1.419b423d5e800p-2, 0x1.8e436ec90e09dp-47},
                          {0x1.7400000000000p-1, 0x1.4718dc271c800p-2, -0x1.f27ce0967d675p-45},
                          {0x1.7200000000000p-1, 0x1.4c9e09e173000p-2, -0x1.e20891b0ad8a4p-45},
                          {0x1.7000000000000p-1, 0x1.522ae0738a000p-2, 0x1.ebe708164c759p-45},
                          {0x1.6e00000000000p-1, 0x1.57bf753c8d000p-2, 0x1.fadedee5d40efp-46},
                          {0x1.6c00000000000p-1, 0x1.5d5bddf596000p-2, -0x1.a0b2a08a465dcp-47},
                      })
// The polynomial of `log(1 + r)` after its linear term, with the scale factors of the evaluation in
// `PowLogarithm` folded in.
CUJU_TABLE_DEFINITION(double, POW_LOGARITHM_POLYNOMIAL,
                      {
                          -0x1p-1,
                          0x1.555555555556p-2 * -2,
                          -0x1.0000000000006p-2 * -2,
                          0x1.999999959554ep-3 * 4,
                          -0x1.555555529a47ap-3 * 4,
                          0x1.2495b9b4845e9p-3 * -8,
                          -0x1.0002b8b263fc3p-3 * -8,
                      })
// Pairs of the tail of `2^(i / 128)` and the bits of its head minus `i << 45`, for i from 0 to 127;
// adding the rounded argument shifted by 45 bits restores the exponent of the head.
CUJU_TABLE_DEFINITION(uint64_t, POW_EXPONENTIAL_TABLE,
                      {
                          0x0ull,
                          0x3ff0000000000000ull,
                          0x3c9b3b4f1a88bf6eull,
                          0x3feff63da9fb3335ull,
                          0xbc7160139cd8dc5dull,
                          0x3fefec9a3e778061ull,
                          0xbc905e7a108766d1ull,
                          0x3fefe315e86e7f85ull,
                          0x3c8cd2523567f613ull,
                          0x3fefd9b0d3158574ull,
                          0xbc8bce8023f98efaull,
                          0x3fefd06b29ddf6deull,
                          0x3c60f74e61e6c861ull,
                          0x3fefc74518759bc8ull,
                          0x3c90a3e45b33d399ull,
                          0x3fefbe3ecac6f383ull,
                          0x3c979aa65d837b6dull,
                          0x3fefb5586cf9890full,
                          0x3c8eb51a92fdeffcull,
                          0x3fefac922b7247f7ull,
                          0x3c3ebe3d702f9cd1ull,
                          0x3fefa3ec32d3d1a2ull,
                          0xbc6a033489906e0bull,
                          0x3fef9b66affed31bull,
                          0xbc9556522a2fbd0eull,
                          0x3fef9301d0125b51ull,
                          0xbc5080ef8c4eea55ull,
                          0x3fef8abdc06c31ccull,
                          0xbc91c923b9d5f416ull,
                          0x3fef829aaea92de0ull,
                          0x3c80d3e3e95c55afull,
                          0x3fef7a98c8a58e51ull,
                          0xbc801b15eaa59348ull,
                          0x3fef72b83c7d517bull,
                          0xbc8f1ff055de323dull,
                          0x3fef6af9388c8deaull,
                          0x3c8b898c3f1353bfull,
                          0x3fef635beb6fcb75ull,
                          0xbc96d99c7611eb26ull,
                          0x3fef5be084045cd4ull,
                          0x3c9aecf73e3a2f60ull,
                          0x3fef54873168b9aaull,
                          0xbc8fe782cb86389dull,
                          0x3fef4d5022fcd91dull,
                          0x3c8a6f4144a6c38dull,
                          0x3fef463b88628cd6ull,
                          0x3c807a05b0e4047dull,
                          0x3fef3f49917ddc96ull,
                          0x3c968efde3a8a894ull,
                          0x3fef387a6e756238ull,
                          0x3c875e18f274487dull,
                          0x3fef31ce4fb2a63full,
                          0x3c80472b981fe7f2ull,
                          0x3fef2b4565e27cddull,
                          0xbc96b87b3f71085eull,
                          0x3fef24dfe1f56381ull,
                          0x3c82f7e16d09ab31ull,
                          0x3fef1e9df51fdee1ull,
                          0xbc3d219b1a6fbffaull,
                          0x3fef187fd0dad990ull,
                          0x3c8b3782720c0ab4ull,
                          0x3fef1285a6e4030bull,
                          0x3c6e149289cecb8full,
                          0x3fef0cafa93e2f56ull,
                          0x3c834d754db0abb6ull,
                          0x3fef06fe0a31b715ull,
                          0x3c864201e2ac744cull,
                          0x3fef0170fc4cd831ull,
                          0x3c8fdd395dd3f84aull,
                          0x3feefc08b26416ffull,
                          0xbc86a3803b8e5b04ull,
                          0x3feef6c55f929ff1ull,
                          0xbc924aedcc4b5068ull,
                          0x3feef1a7373aa9cbull,
                          0xbc9907f81b512d8eull,
                          0x3feeecae6d05d866ull,
                          0xbc71d1e83e9436d2ull,
                          0x3feee7db34e59ff7ull,
                          0xbc991919b3ce1b15ull,
                          0x3feee32dc313a8e5ull,
                          0x3c859f48a72a4c6dull,
                          0x3feedea64c123422ull,
                          0xbc9312607a28698aull,
                          0x3feeda4504ac801cull,
                          0xbc58a78f4817895bull,
                          0x3feed60a21f72e2aull,
                          0xbc7c2c9b67499a1bull,
                          0x3feed1f5d950a897ull,
                          0x3c4363ed60c2ac11ull,
                          0x3feece086061892dull,
                          0x3c9666093b0664efull,
                          0x3feeca41ed1d0057ull,
                          0x3c6ecce1daa10379ull,
                          0x3feec6a2b5c13cd0ull,
                          0x3c93ff8e3f0f1230ull,
                          0x3feec32af0d7d3deull,
                          0x3c7690cebb7aafb0ull,
                          0x3feebfdad5362a27ull,
                          0x3c931dbdeb54e077ull,
                          0x3feebcb299fddd0dull,
                          0xbc8f94340071a38eull,
                          0x3feeb9b2769d2ca7ull,
                          0xbc87deccdc93a349ull,
                          0x3feeb6daa2cf6642ull,
                          0xbc78dec6bd0f385full,
                          0x3feeb42b569d4f82ull,
                          0xbc861246ec7b5cf6ull,
                          0x3feeb1a4ca5d920full,
                          0x3c93350518fdd78eull,
                          0x3feeaf4736b527daull,
                          0x3c7b98b72f8a9b05ull,
                          0x3feead12d497c7fdull,
                          0x3c9063e1e21c5409ull,
                          0x3feeab07dd485429ull,
                          0x3c34c7855019c6eaull,
                          0x3feea9268a5946b7ull,
                          0x3c9432e62b64c035ull,
                          0x3feea76f15ad2148ull,
                          0xbc8ce44a6199769full,
                          0x3feea5e1b976dc09ull,
                          0xbc8c33c53bef4da8ull,
                          0x3feea47eb03a5585ull,
                          0xbc845378892be9aeull,
                          0x3feea34634ccc320ull,
                          0xbc93cedd78565858ull,
                          0x3feea23882552225ull,
                          0x3c5710aa807e1964ull,
                          0x3feea155d44ca973ull,
                          0xbc93b3efbf5e2228ull,
                          0x3feea09e667f3bcdull,
                          0xbc6a12ad8734b982ull,
                          0x3feea012750bdabfull,
                          0xbc6367efb86da9eeull,
                          0x3fee9fb23c651a2full,
                          0xbc80dc3d54e08851ull,
                          0x3fee9f7df9519484ull,
                          0xbc781f647e5a3ecfull,
                          0x3fee9f75e8ec5f74ull,
                          0xbc86ee4ac08b7db0ull,
                          0x3fee9f9a48a58174ull,
                          0xbc8619321e55e68aull,
                          0x3fee9feb564267c9ull,
                          0x3c909ccb5e09d4d3ull,
                          0x3feea0694fde5d3full,
                          0xbc7b32dcb94da51dull,
                          0x3feea11473eb0187ull,
                          0x3c94ecfd5467c06bull,
                          0x3feea1ed0130c132ull,
                          0x3c65ebe1abd66c55ull,
                          0x3feea2f336cf4e62ull,
                          0xbc88a1c52fb3cf42ull,
                          0x3feea427543e1a12ull,
                          0xbc9369b6f13b3734ull,
                          0x3feea589994cce13ull,
                          0xbc805e843a19ff1eull,
                          0x3feea71a4623c7adull,
                          0xbc94d450d872576eull,
                          0x3feea8d99b4492edull,
                          0x3c90ad675b0e8a00ull,
                          0x3feeaac7d98a6699ull,
                          0x3c8db72fc1f0eab4ull,
                          0x3feeace5422aa0dbull,
                          0xbc65b6609cc5e7ffull,
                          0x3feeaf3216b5448cull,
                          0x3c7bf68359f35f44ull,
                          0x3feeb1ae99157736ull,
                          0xbc93091fa71e3d83ull,
                          0x3feeb45b0b91ffc6ull,
                          0xbc5da9b88b6c1e29ull,
                          0x3feeb737b0cdc5e5ull,
                          0xbc6c23f97c90b959ull,
                          0x3feeba44cbc8520full,
                          0xbc92434322f4f9aaull,
                          0x3feebd829fde4e50ull,
                          0xbc85ca6cd7668e4bull,
                          0x3feec0f170ca07baull,
                          0x3c71affc2b91ce27ull,
                          0x3feec49182a3f090ull,
                          0x3c6dd235e10a73bbull,
                          0x3feec86319e32323ull,
                          0xbc87c50422622263ull,
                          0x3feecc667b5de565ull,
                          0x3c8b1c86e3e231d5ull,
                          0x3feed09bec4a2d33ull,
                          0xbc91bbd1d3bcbb15ull,
                          0x3feed503b23e255dull,
                          0x3c90cc319cee31d2ull,
                          0x3feed99e1330b358ull,
                          0x3c8469846e735ab3ull,
                          0x3feede6b5579fdbfull,
                          0xbc82dfcd978e9db4ull,
                          0x3feee36bbfd3f37aull,
                          0x3c8c1a7792cb3387ull,
                          0x3feee89f995ad3adull,
                          0xbc907b8f4ad1d9faull,
                          0x3feeee07298db666ull,
                          0xbc55c3d956dcaebaull,
                          0x3feef3a2b84f15fbull,
                          0xbc90a40e3da6f640ull,
                          0x3feef9728de5593aull,
                          0xbc68d6f438ad9334ull,
                          0x3feeff76f2fb5e47ull,
                          0xbc91eee26b588a35ull,
                          0x3fef05b030a1064aull,
                          0x3c74ffd70a5fddcdull,
                          0x3fef0c1e904bc1d2ull,
                          0xbc91bdfbfa9298acull,
                          0x3fef12c25bd71e09ull,
                          0x3c736eae30af0cb3ull,
                          0x3fef199bdd85529cull,
                          0x3c8ee3325c9ffd94ull,
                          0x3fef20ab5fffd07aull,
                          0x3c84e08fd10959acull,
                          0x3fef27f12e57d14bull,
                          0x3c63cdaf384e1a67ull,
                          0x3fef2f6d9406e7b5ull,
                          0x3c676b2c6c921968ull,
                          0x3fef3720dcef9069ull,
                          0xbc808a1883ccb5d2ull,
                          0x3fef3f0b555dc3faull,
                          0xbc8fad5d3ffffa6full,
                          0x3fef472d4a07897cull,
                          0xbc900dae3875a949ull,
                          0x3fef4f87080d89f2ull,
                          0x3c74a385a63d07a7ull,
                          0x3fef5818dcfba487ull,
                          0xbc82919e2040220full,
                          0x3fef60e316c98398ull,
                          0x3c8e5a50d5c192acull,
                          0x3fef69e603db3285ull,
                          0x3c843a59ac016b4bull,
                          0x3fef7321f301b460ull,
                          0xbc82d52107b43e1full,
                          0x3fef7c97337b9b5full,
                          0xbc892ab93b470dc9ull,
                          0x3fef864614f5a129ull,
                          0x3c74b604603a88d3ull,
                          0x3fef902ee78b3ff6ull,
                          0x3c83c5ec519d7271ull,
                          0x3fef9a51fbc74c83ull,
                          0xbc8ff7128fd391f0ull,
                          0x3fefa4afa2a490daull,
                          0xbc8dae98e223747dull,
                          0x3fefaf482d8e67f1ull,
                          0x3c8ec3bc41aa2008ull,
                          0x3fefba1bee615a27ull,
                          0x3c842b94c3a9eb32ull,
                          0x3fefc52b376bba97ull,
                          0x3c8a64a931d185eeull,
                          0x3fefd0765b6e4540ull,
                          0xbc8e37bae43be3edull,
                          0x3fefdbfdad9cbe14ull,
                          0x3c77893b4d91cd9dull,
                          0x3fefe7c1819e90d8ull,
                          0x3c5305c14160cc89ull,
                          0x3feff3c22b8f71f1ull,
                      })
// The polynomial of `exp(r) - 1 - r` from the square term up.
CUJU_TABLE_DEFINITION(double, POW_EXPONENTIAL_POLYNOMIAL,
                      {
                          0x1.ffffffffffdbdp-2,
                          0x1.555555555543cp-3,
                          0x1.55555cf172b91p-5,
                          0x1.1111167a4d017p-7,
                      })

constexpr double POW_LOGARITHM_OF_TWO_HIGH = 0x1.62e42fefa3800p-1;
constexpr double POW_LOGARITHM_OF_TWO_LOW = 0x1.ef35793c76730p-45;
constexpr double POW_EXPONENTIAL_STEPS_PER_UNIT = 0x1.71547652b82fep0 * 128;
// The high and low parts of `-log(2) / 128`, the negated length of one table step.
constexpr double POW_EXPONENTIAL_NEGATED_STEP_HIGH = -0x1.62e42fefa0000p-8;
constexpr double POW_EXPONENTIAL_NEGATED_STEP_LOW = -0x1.cf79abc9e3b3ap-47;
// 1.5 * 2^52: the sum with a double rounds it to an integer.
constexpr double POW_EXPONENTIAL_ROUND_SHIFT = 0x1.8p52;

// pow.c's `top12`: the sign and the exponent field.
CUJU_HOST_DEVICE uint32_t PowTopBits(double value) { return (uint32_t)(DoubleBits(value) >> 52); }

// pow.c's `log_inline`: returns the head of the logarithm and leaves its tail in `*tail`. `bits`
// holds a positive normal double, or a subnormal double times 2^52 with 52 taken from its exponent
// field.
CUJU_HOST_DEVICE double PowLogarithm(uint64_t bits, double *tail) {
    const PowLogarithmEntry *table = CUJU_TABLE(POW_LOGARITHM_TABLE);
    const double *coefficients = CUJU_TABLE(POW_LOGARITHM_POLYNOMIAL);
    // The offset centers the subintervals on 1, so that `bits - offset` gives the exponent and the
    // subinterval of the argument.
    constexpr uint64_t offset = 0x3fe6955500000000ull;
    uint64_t temporary = bits - offset;
    int index = (int)((temporary >> (52 - 7)) % 128);
    int exponent = (int)((int64_t)temporary >> 52);
    uint64_t normalizedBits = bits - (temporary & 0xfffull << 52);
    double normalized = DoubleFromBits(normalizedBits);
    double exponentValue = (double)exponent;
    double inverseCenter = table[index].inverseCenter,
           logarithmOfCenter = table[index].logarithmOfCenter,
           logarithmOfCenterTail = table[index].logarithmOfCenterTail;
    double reduced = fma(normalized, inverseCenter, -1.0);
    double high1 = fma(exponentValue, POW_LOGARITHM_OF_TWO_HIGH, logarithmOfCenter);
    double high2 = high1 + reduced;
    double low1 = fma(exponentValue, POW_LOGARITHM_OF_TWO_LOW, logarithmOfCenterTail);
    double low2 = high1 - high2 + reduced;
    double halfTerm = coefficients[0] * reduced;
    double quadraticTerm = reduced * halfTerm;
    double cubicTerm = reduced * quadraticTerm;
    double high = high2 + quadraticTerm;
    double low3 = fma(halfTerm, reduced, -quadraticTerm);
    double low4 = high2 - high + quadraticTerm;
    // The FMA build contracts `cubicTerm * (A1 + r * A2 + q * (A3 + r * A4 + q * (A5 + r * A6)))`,
    // with `A` the coefficients, `r` `reduced` and `q` `quadraticTerm`, and fuses its last product
    // into the sum `low1 + low2 + low3 + low4 + polynomial`.
    double series56 = fma(reduced, coefficients[6], coefficients[5]);
    double series34 = fma(quadraticTerm, series56, fma(reduced, coefficients[4], coefficients[3]));
    double series12 = fma(quadraticTerm, series34, fma(reduced, coefficients[2], coefficients[1]));
    double low = fma(cubicTerm, series12, low1 + low2 + low3 + low4);
    double result = high + low;
    *tail = high - result + low;
    return result;
}

// math_err.c's `__math_oflow` and `__math_uflow`: the product overflows or underflows on purpose,
// to raise that exception.
CUJU_HOST_DEVICE double PowOverflow(uint32_t sign) {
    double value = sign ? -0x1p769 : 0x1p769;
    return value * 0x1p769;
}
CUJU_HOST_DEVICE double PowUnderflow(uint32_t sign) {
    double value = sign ? -0x1p-767 : 0x1p-767;
    return value * 0x1p-767;
}

// pow.c's `specialcase`: `scale * (1 + temporary)` for a scale that overflows or underflows on its
// own. Bit 31 of `roundedBits` marks a negative argument.
CUJU_HOST_DEVICE_NO_INLINE double PowScaleSpecialCase(double temporary, uint64_t scaleBits,
                                                      uint64_t roundedBits) {
    double scale, result;
    if ((roundedBits & 0x80000000) == 0) {
        // The scale overflows, so the steps use it divided by 2^1009 and multiply by 2^1009 last.
        scaleBits -= 1009ull << 52;
        scale = DoubleFromBits(scaleBits);
        result = 0x1p1009 * fma(scale, temporary, scale);
        return result;
    }
    // The scale underflows, so the steps use it times 2^1022 and multiply by 2^-1022 last.
    scaleBits += 1022ull << 52;
    scale = DoubleFromBits(scaleBits);
    // Both `result` and `low` read this product, so it stays unfused.
    double scaledTemporary = scale * temporary;
    result = scale + scaledTemporary;
    if (fabs(result) < 1.0) {
        // A result below 1 in magnitude rounds again when the final product makes it subnormal.
        // Adding and removing 1 moves that rounding into this step, so the result rounds once.
        double high, low, one = 1.0;
        if (result < 0.0) one = -1.0;
        low = scale - result + scaledTemporary;
        high = one + result;
        low = one - high + result + low;
        result = (high + low) - one;
        if (result == 0.0) result = DoubleFromBits(scaleBits & 0x8000000000000000ull);
    }
    return 0x1p-1022 * result;
}

// pow.c's `exp_inline`: `exp(argument + argumentTail)`, with the sign bit that `signBias` sets when
// it is not 0.
CUJU_HOST_DEVICE double PowExponential(double argument, double argumentTail, uint32_t signBias) {
    const uint64_t *table = CUJU_TABLE(POW_EXPONENTIAL_TABLE);
    const double *coefficients = CUJU_TABLE(POW_EXPONENTIAL_POLYNOMIAL);
    uint32_t magnitudeTop = PowTopBits(argument) & 0x7ff;
    if (magnitudeTop - PowTopBits(0x1p-54) >= PowTopBits(512.0) - PowTopBits(0x1p-54)) {
        // An argument below 2^-54 in magnitude gives `1 + argument`.
        if (magnitudeTop - PowTopBits(0x1p-54) >= 0x80000000) {
            double one = 1.0 + argument;
            return signBias ? -one : one;
        }
        if (magnitudeTop >= PowTopBits(1024.0)) {
            if (DoubleBits(argument) >> 63) return PowUnderflow(signBias);
            return PowOverflow(signBias);
        }
        // An argument from 512 to 1024 in magnitude goes through `PowScaleSpecialCase`.
        magnitudeTop = 0;
    }
    // One fused step computes `steps = POW_EXPONENTIAL_STEPS_PER_UNIT * argument` and
    // `rounded = steps + POW_EXPONENTIAL_ROUND_SHIFT`.
    double rounded = fma(POW_EXPONENTIAL_STEPS_PER_UNIT, argument, POW_EXPONENTIAL_ROUND_SHIFT);
    uint64_t roundedBits = DoubleBits(rounded);
    rounded -= POW_EXPONENTIAL_ROUND_SHIFT;
    double reduced = fma(rounded, POW_EXPONENTIAL_NEGATED_STEP_LOW,
                         fma(rounded, POW_EXPONENTIAL_NEGATED_STEP_HIGH, argument));
    reduced += argumentTail;
    uint64_t index = 2 * (roundedBits % 128);
    uint64_t top = (roundedBits + signBias) << (52 - 7);
    double tail = DoubleFromBits(table[index]);
    uint64_t scaleBits = table[index + 1] + top;
    double reducedSquare = reduced * reduced;
    // The compiler contracts `tail + r + r^2 * (c0 + r * c1) + r^4 * (c2 + r * c3)` into these
    // fused steps. `c` stands for `coefficients` and `r` for `reduced`.
    double temporary =
        fma(fma(reduced, coefficients[3], coefficients[2]), reducedSquare * reducedSquare,
            fma(fma(reduced, coefficients[1], coefficients[0]), reducedSquare, tail + reduced));
    if (magnitudeTop == 0) return PowScaleSpecialCase(temporary, scaleBits, roundedBits);
    double scale = DoubleFromBits(scaleBits);
    return fma(scale, temporary, scale);
}

// pow.c's `checkint` for a nonzero finite double: 0 if it is not an integer, 1 for an odd integer
// and 2 for an even one.
CUJU_HOST_DEVICE int PowIntegerKind(uint64_t bits) {
    int exponent = bits >> 52 & 0x7ff;
    if (exponent < 0x3ff) return 0;
    if (exponent > 0x3ff + 52) return 2;
    if (bits & ((1ull << (0x3ff + 52 - exponent)) - 1)) return 0;
    if (bits & (1ull << (0x3ff + 52 - exponent))) return 1;
    return 2;
}

CUJU_HOST_DEVICE int PowIsZeroInfinityOrNaN(uint64_t bits) {
    return 2 * bits - 1 >= 2 * DoubleBits(INFINITY) - 1;
}

CUJU_HOST_DEVICE bool IsSignalingDouble(double value) {
    // A clear top bit of the significand marks a signaling NaN on x86.
    uint64_t bits = DoubleBits(value);
    return 2 * (bits ^ 0x0008000000000000ull) > 2 * 0x7ff8000000000000ull;
}

CUJU_HOST_DEVICE_NO_INLINE double pow(double base, double exponent) {
    // The bias sets the sign bit of the result once `PowExponential` shifts it into place.
    constexpr uint32_t SIGN_BIAS = 0x800 << 7;
    uint32_t signBias = 0;
    uint64_t baseBits = DoubleBits(base), exponentBits = DoubleBits(exponent);
    uint32_t baseTop = PowTopBits(base), exponentTop = PowTopBits(exponent);
    if (baseTop - 0x001 >= 0x7ff - 0x001 || (exponentTop & 0x7ff) - 0x3be >= 0x43e - 0x3be) {
        if (PowIsZeroInfinityOrNaN(exponentBits)) {
            if (2 * exponentBits == 0) return IsSignalingDouble(base) ? base + exponent : 1.0;
            if (baseBits == DoubleBits(1.0))
                return IsSignalingDouble(exponent) ? base + exponent : 1.0;
            if (2 * baseBits > 2 * DoubleBits(INFINITY) ||
                2 * exponentBits > 2 * DoubleBits(INFINITY))
                return base + exponent;
            if (2 * baseBits == 2 * DoubleBits(1.0)) return 1.0;
            if ((2 * baseBits < 2 * DoubleBits(1.0)) == !(exponentBits >> 63)) return 0.0;
            return exponent * exponent;
        }
        if (PowIsZeroInfinityOrNaN(baseBits)) {
            double baseSquare = base * base;
            if (baseBits >> 63 && PowIntegerKind(exponentBits) == 1) baseSquare = -baseSquare;
            return exponentBits >> 63 ? 1 / baseSquare : baseSquare;
        }
        if (baseBits >> 63) {
            int integerKind = PowIntegerKind(exponentBits);
            if (integerKind == 0) return (base - base) / (base - base);
            if (integerKind == 1) signBias = SIGN_BIAS;
            baseBits &= 0x7fffffffffffffffull;
            baseTop &= 0x7ff;
        }
        // An exponent below 2^-65 or from 2^63 up in magnitude gives a result near 1, an infinity
        // or a zero.
        if ((exponentTop & 0x7ff) - 0x3be >= 0x43e - 0x3be) {
            if (baseBits == DoubleBits(1.0)) return 1.0;
            if ((exponentTop & 0x7ff) < 0x3be)
                return baseBits > DoubleBits(1.0) ? 1.0 + exponent : 1.0 - exponent;
            return (baseBits > DoubleBits(1.0)) == (exponentTop < 0x800) ? PowOverflow(0)
                                                                         : PowUnderflow(0);
        }
        // The product with 2^52 makes a subnormal base normal, and the subtraction takes the 52
        // back from its exponent field.
        if (baseTop == 0) {
            baseBits = DoubleBits(base * 0x1p52);
            baseBits &= 0x7fffffffffffffffull;
            baseBits -= 52ull << 52;
        }
    }
    double low;
    double high = PowLogarithm(baseBits, &low);
    double productHigh = exponent * high;
    double productLow = fma(exponent, low, fma(exponent, high, -productHigh));
    return PowExponential(productHigh, productLow, signBias);
}

}  // namespace libm
}  // namespace cuju
