import sys, traceback
sys.path.insert(0, '.')
try:
    from database.db import init_db, SessionLocal
    from database.models import Location, Person, Sighting, ObjectSighting
    init_db()
    db = SessionLocal()
    import json, uuid
    from datetime import datetime
    p = Person(id=str(uuid.uuid4()), unique_code='SDT-DEBUG1',
               face_embedding=json.dumps([0.0]*10),
               created_at=datetime.utcnow(),
               entry_zone='Entrance', location_id='LOC-001', person_type='visitor')
    db.add(p)
    db.commit()
    print('Person added OK')
    s = Sighting(id=str(uuid.uuid4()), person_id=p.id,
                 location_id='LOC-001', zone_id='Entrance',
                 camera_id='CAM-001', seen_at=datetime.utcnow(),
                 confidence=0.9)
    db.add(s)
    db.commit()
    print('Sighting added OK')
    # cleanup
    db.delete(s); db.delete(p); db.commit()
    db.close()
    print('ALL OK')
except Exception:
    traceback.print_exc()
    sys.exit(1)
