from liga.oiv4.metadata import OIV4MetadataProvider


def test_that_implied_objects_are_correct():
    oiv4_meta = OIV4MetadataProvider()
    food_id = oiv4_meta.object_names.index('Food')
    seafood_id = oiv4_meta.object_names.index('Seafood')
    squid_id = oiv4_meta.object_names.index('Squid')
    baked_goods_id = oiv4_meta.object_names.index('Baked goods')
    bread_id = oiv4_meta.object_names.index('Bread')

    assert food_id in oiv4_meta.get_implied_objects(seafood_id)
    assert seafood_id not in oiv4_meta.get_implied_objects(food_id)
    assert food_id in oiv4_meta.get_implied_objects(squid_id)
    assert seafood_id in oiv4_meta.get_implied_objects(squid_id)
    assert squid_id not in oiv4_meta.get_implied_objects(food_id)

    assert baked_goods_id in oiv4_meta.get_implied_objects(bread_id)
    assert baked_goods_id not in oiv4_meta.get_implied_objects(squid_id)
