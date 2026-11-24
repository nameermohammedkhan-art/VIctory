//! End-to-end tests of waterfall loops over per-lane descriptors: the
//! synthetic GFX10 programs of `resource::tests` lowered to DXIL, validated
//! by `dxil.dll` when present, and run on the host GPU and WARP, each lane
//! reading the resource its own descriptor names.

use windows::Win32::Graphics::Direct3D12::*;

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// Lowers a wave32 compute program of 32 threads with the table of views at
/// s0 (textures, or with `buffers` structured buffers of dwords, at entries
/// 0 and 1, a raw UAV at entry 2) and the table of samplers at s1.
fn build(words: &[u32], buffers: bool) -> HostShader {
    let table = |user_data: u8| RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: user_data,
    };
    let range = |kind, table_offset_dwords, dwords, stride, user_data| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords,
        user_data,
        slot: user_data,
    };
    let views = if buffers {
        range(ResourceType::Srv, 0, 4, 4, 0)
    } else {
        range(ResourceType::Srv, 0, 8, 0, 0)
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![table(0), table(1)],
        used: vec![
            views,
            range(ResourceType::Uav, 16, 4, 0, 0),
            range(ResourceType::Sampler, 0, 4, 0, 1),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");
    // One access with each lane's own heap index, no wave operation.
    assert_eq!(lowered.nonuniform_handles, 1);
    assert!(!lowered.wave_ops);
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: Vec::new(),
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

/// Runs `shader` with `views` at heap indices 0.. (the UAV last) on the
/// hardware GPU and on WARP (each skipped without a device) and returns,
/// per one that ran, the UAV's 32 dwords.
fn run(shader: &HostShader, views: &[View]) -> Vec<(&'static str, Vec<u32>)> {
    let samplers = [D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        MaxLOD: f32::MAX,
        ..Default::default()
    }];
    let mut out = Vec::new();
    // One GPU test at a time, with the VALU tests: in parallel on the
    // hardware adapter and WARP the test process sometimes died of an
    // access violation (7 of 24 runs of these four tests).
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = match host.compute_pipeline(shader) {
            Ok(p) => p,
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        // Both tables start at heap index 0.
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            assert_eq!(u.root, RootType::Table, "{u:?}");
            ud[u.index as usize] = 0;
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views,
                    samplers: &samplers,
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, words[0].clone()));
    }
    out
}

/// The red channel each lane sampled from two 1x1 textures, red 1.0 at
/// heap index 0 and 2.0 at 1.
fn sampled(shader: &HostShader) -> Vec<(&'static str, Vec<f32>)> {
    let first = [[1.0f32, 0.0, 0.0, 1.0]];
    let second = [[2.0f32, 0.0, 0.0, 1.0]];
    let views = [
        View::Texture(1, 1, &first),
        View::Texture(1, 1, &second),
        View::RawUav(32),
    ];
    run(shader, &views)
        .into_iter()
        .map(|(label, words)| (label, words.into_iter().map(f32::from_bits).collect()))
        .collect()
}

#[test]
fn lanes_sample_the_texture_their_if_else_arm_chose() {
    use crate::resource::tests::WATERFALL_JOINED_WAVE32;
    let shader = build(WATERFALL_JOINED_WAVE32, false);
    for (label, red) in sampled(&shader) {
        // Lanes below 16 took the first texture, the others the second.
        let want: Vec<f32> = (0..32).map(|l| if l < 16 { 1.0 } else { 2.0 }).collect();
        assert_eq!(red, want, "{label}");
    }
}

#[test]
fn lanes_sample_the_texture_their_arm_loaded_or_picked() {
    use crate::resource::tests::WATERFALL_JOINED_MIXED_WAVE32;
    let shader = build(WATERFALL_JOINED_MIXED_WAVE32, false);
    for (label, red) in sampled(&shader) {
        // Lanes below 16 loaded entry (lane & 1); of the others, lanes
        // below 24 picked the first, the rest the second.
        let want: Vec<f32> = (0..32)
            .map(|l| match l {
                0..16 => [1.0, 2.0][l & 1],
                16..24 => 1.0,
                _ => 2.0,
            })
            .collect();
        assert_eq!(red, want, "{label}");
    }
}

#[test]
fn lanes_load_from_the_buffer_their_nested_arms_chose() {
    use crate::resource::tests::WATERFALL_JOINED_BUFFER_WAVE32;
    let shader = build(WATERFALL_JOINED_BUFFER_WAVE32, true);
    let first: Vec<u32> = (0..32).map(|i| 100 + i).collect();
    let second: Vec<u32> = (0..32).map(|i| 200 + i).collect();
    let views = [
        View::Buffer {
            data: &first,
            stride: Stride::Structured(4),
        },
        View::Buffer {
            data: &second,
            stride: Stride::Structured(4),
        },
        View::RawUav(32),
    ];
    for (label, got) in run(&shader, &views) {
        // Dword `lane` of the first buffer below 16, of the second from 16
        // to 23; lanes from 24 on load and store nothing.
        let want: Vec<u32> = (0..32)
            .map(|l| match l {
                0..16 => 100 + l,
                16..24 => 200 + l,
                _ => 0,
            })
            .collect();
        assert_eq!(got, want, "{label}");
    }
}

#[test]
fn lanes_sample_the_entry_their_offset_names_in_the_console_waterfall() {
    use crate::resource::tests::WATERFALL_INDEXED_WAVE32;
    let shader = build(WATERFALL_INDEXED_WAVE32, false);
    let results = sampled(&shader);
    assert!(!results.is_empty());
    for (label, red) in results {
        // Even lanes hold entry 0's offset, odd lanes entry 1's.
        let want: Vec<f32> = (0..32).map(|l| [1.0, 2.0][l & 1]).collect();
        assert_eq!(red, want, "{label}");
    }
}

#[test]
fn lanes_sample_their_own_texture_in_a_waterfall_narrowed_by_v_cmpx() {
    use crate::resource::tests::cmpx_waterfall;
    // Lanes below 16 hold the first descriptor, the others the second.
    let want: Vec<f32> = (0..32).map(|l| if l < 16 { 1.0 } else { 2.0 }).collect();
    for folded in [false, true] {
        let shader = build(&cmpx_waterfall(false, folded), false);
        let results = sampled(&shader);
        assert!(!results.is_empty());
        for (label, red) in results {
            assert_eq!(red, want, "{label}, folded {folded}");
        }
    }
}

#[test]
fn immediate_samplers_chosen_across_control_flow_sample_expected_mode() {
    // Compute wave32: s0 is a uniform root constant (0 = WRAP, nonzero = CLAMP);
    // s1 is a descriptor table holding a 2x1 texture SRV at offset 0 (texels 1.0, 2.0)
    // and a raw UAV at offset 0x40 (heap index 2). Samples at u = 1.25, v = 0.5:
    // WRAP wraps 1.25 -> 0.25 (texel 0 = 1.0); CLAMP clamps 1.25 -> 1.0 (texel 1 = 2.0).
    let words: &[u32] = &[
        0xBE82_0301, // s_mov_b32 s2, s1
        0xBE83_0380, // s_mov_b32 s3, 0
        0xF40C_0201,
        0xFA00_0000, // s_load_dwordx8 s[8:15], s[2:3], 0x0
        0xF408_0401,
        0xFA00_0040, // s_load_dwordx4 s[16:19], s[2:3], 0x40
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF85_0006, // s_cbranch_scc1 -> wrap_arm (+6 dwords)
        // clamp_arm (6 dwords): address U=V=W=2 (CLAMP), point filter
        0xBE84_03FF,
        0x0000_0092, // s_mov_b32 s4, 0x92
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0380, // s_mov_b32 s6, 0
        0xBE87_0380, // s_mov_b32 s7, 0
        0xBF82_0004, // s_branch -> merge (+4 dwords)
        // wrap_arm (4 dwords): address U=V=W=0 (WRAP), point filter
        0xBE84_0380, // s_mov_b32 s4, 0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0380, // s_mov_b32 s6, 0
        0xBE87_0380, // s_mov_b32 s7, 0
        // merge: sample at (1.25, 0.5) and store lane's red channel to UAV[lane]
        0x7E04_02FF,
        0x3FA0_0000, // v_mov_b32 v2, 1.25f
        0x7E06_02FF,
        0x3F00_0000, // v_mov_b32 v3, 0.5f
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xF09C_0F08,
        0x0022_0C02, // image_sample_lz v[12:15], v[2:3], s[8:15], s[4:7] dmask:0xf dim:2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1000,
        0x8004_0C10, // buffer_store_dword v12, v16, s[16:19], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 1,
                slot: 1,
            },
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 8,
                user_data: 1,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 16,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 1,
                slot: 1,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");
    assert_eq!(lowered.samplers.len(), 2);
    let static_samplers = lowered
        .samplers
        .iter()
        .map(|&w| super::StaticSampler::from_words(w))
        .collect();
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers,
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_indexed_views: Vec::new(),
        root_uavs: lowered.root_uavs,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let texels = [[1.0f32, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]];
    let dummy = [[0.0f32, 0.0, 0.0, 1.0]];
    let views = [
        View::Texture(2, 1, &texels),
        View::Texture(1, 1, &dummy),
        View::RawUav(32),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (selector, expected) in [(0u32, 1.0f32), (1u32, 2.0f32)] {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match u.root {
                    RootType::Constants => selector,
                    RootType::Table => 0,
                    _ => 0,
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label} sel={selector}: dispatch failed: {e}"));
            let got: Vec<f32> = out[0].iter().copied().map(f32::from_bits).collect();
            assert_eq!(got, vec![expected; 32], "{label} selector={selector}");
        }
    }
}

#[test]
fn nested_loaded_descriptor_phi_validates_and_runs() {
    // Nested uniform control-flow choice of three table SRVs (entries 0, 1, 2)
    // producing a `Source::Phi` containing a nested `Source::Phi`:
    // s0 == 0 -> entry 0 (1.0); else if s1 == 0 -> entry 1 (2.0); else -> entry 2 (3.0).
    let words: &[u32] = &[
        0xBE84_0302, // s_mov_b32 s4, s2 (table)
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0402,
        0xFA00_0060, // s_load_dwordx4 s[16:19], s[4:5], 0x60 (UAV at entry 3)
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF85_0009, // s_cbranch_scc1 -> arm0 (+9 dwords)
        // inner_cond (2 dwords):
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF85_0003, // s_cbranch_scc1 -> arm1 (+3 dwords)
        // arm2 (3 dwords):
        0xF40C_0202,
        0xFA00_0040, // s_load_dwordx8 s[8:15], s[4:5], 0x40 (entry 2)
        0xBF82_0002, // s_branch -> inner_join (+2 dwords)
        // arm1 (2 dwords):
        0xF40C_0202,
        0xFA00_0020, // s_load_dwordx8 s[8:15], s[4:5], 0x20 (entry 1)
        // inner_join (2 dwords) — produces inner Source::Phi:
        0xBF80_0000, // s_nop 0
        0xBF82_0002, // s_branch -> outer_join (+2 dwords)
        // arm0 (2 dwords):
        0xF40C_0202,
        0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0x0 (entry 0)
        // outer_join — produces outer Source::Phi over inner Source::Phi and arm0:
        0xBE94_0380, // s_mov_b32 s20, 0
        0xBE95_0380, // s_mov_b32 s21, 0
        0xBE96_0380, // s_mov_b32 s22, 0
        0xBE97_0380, // s_mov_b32 s23, 0
        0x7E04_02FF,
        0x3F00_0000, // v_mov_b32 v2, 0.5f
        0x7E06_02FF,
        0x3F00_0000, // v_mov_b32 v3, 0.5f
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xF09C_0F08,
        0x00A2_0C02, // image_sample_lz v[12:15], v[2:3], s[8:15], s[20:23] dmask:0xf dim:2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1000,
        0x8004_0C10, // buffer_store_dword v12, v16, s[16:19], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 2,
                slot: 1,
            },
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 8,
                user_data: 2,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 24,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 2,
                slot: 1,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 3 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    assert!(
        res.descriptors.iter().any(|d| matches!(
            &d.source,
            resource::Source::Phi { args, .. }
                if args.iter().any(|a| matches!(a, resource::Source::Phi { .. }))
        )),
        "expected a nested Source::Phi: {:?}",
        res.descriptors
    );
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");
    let static_samplers = lowered
        .samplers
        .iter()
        .map(|&w| super::StaticSampler::from_words(w))
        .collect();
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers,
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_indexed_views: Vec::new(),
        root_uavs: lowered.root_uavs,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let t0 = [[1.0f32, 0.0, 0.0, 1.0]];
    let t1 = [[2.0f32, 0.0, 0.0, 1.0]];
    let t2 = [[3.0f32, 0.0, 0.0, 1.0]];
    let views = [
        View::Texture(1, 1, &t0),
        View::Texture(1, 1, &t1),
        View::Texture(1, 1, &t2),
        View::RawUav(32),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (c0, c1, expected) in [(0u32, 0u32, 1.0f32), (1, 0, 2.0), (1, 1, 3.0)] {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.dword) {
                    (RootType::Constants, 0) => c0,
                    (RootType::Constants, 1) => c1,
                    _ => 0,
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label} ({c0},{c1}): dispatch failed: {e}"));
            let got: Vec<f32> = out[0].iter().copied().map(f32::from_bits).collect();
            assert_eq!(got, vec![expected; 32], "{label} ({c0},{c1})");
        }
    }
}
