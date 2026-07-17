-- docker/init.sql
-- SmartDetect — Universal Camera Detection System
-- Auto-executed by Docker Compose on first postgres startup.
-- ─────────────────────────────────────────────────────────────────────────────

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Locations table (replaces stations)
CREATE TABLE IF NOT EXISTS locations (
    id         VARCHAR(64)  PRIMARY KEY,
    name       VARCHAR(128) NOT NULL,
    type       VARCHAR(64)  NOT NULL DEFAULT 'other',
    address    VARCHAR(256),
    created_at TIMESTAMP    NOT NULL DEFAULT NOW()
);

-- 3. Persons table
CREATE TABLE IF NOT EXISTS persons (
    id             VARCHAR(36)  PRIMARY KEY DEFAULT gen_random_uuid()::text,
    unique_code    VARCHAR(32)  NOT NULL UNIQUE,
    face_embedding TEXT,
    created_at     TIMESTAMP    NOT NULL DEFAULT NOW(),
    entry_zone     VARCHAR(64),
    location_id    VARCHAR(64)  REFERENCES locations(id) ON DELETE SET NULL,
    person_type    VARCHAR(32)  NOT NULL DEFAULT 'unknown'
);
CREATE INDEX IF NOT EXISTS idx_persons_unique_code ON persons (unique_code);

-- 4. Sightings table
CREATE TABLE IF NOT EXISTS sightings (
    id                   VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
    person_id            VARCHAR(36) NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    location_id          VARCHAR(64) REFERENCES locations(id) ON DELETE SET NULL,
    zone_id              VARCHAR(64),
    camera_id            VARCHAR(64) NOT NULL,
    seen_at              TIMESTAMP   NOT NULL DEFAULT NOW(),
    confidence           FLOAT       NOT NULL,
    frame_snapshot_path  TEXT
);
CREATE INDEX IF NOT EXISTS idx_sightings_person_id  ON sightings (person_id);
CREATE INDEX IF NOT EXISTS idx_sightings_location_id ON sightings (location_id);
CREATE INDEX IF NOT EXISTS idx_sightings_seen_at    ON sightings (seen_at);

-- 5. Object sightings table (YOLOv8)
CREATE TABLE IF NOT EXISTS object_sightings (
    id                   VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
    location_id          VARCHAR(64) REFERENCES locations(id) ON DELETE SET NULL,
    zone_id              VARCHAR(64),
    camera_id            VARCHAR(64) NOT NULL,
    object_type          VARCHAR(64) NOT NULL,
    confidence           FLOAT       NOT NULL,
    detected_at          TIMESTAMP   NOT NULL DEFAULT NOW(),
    bbox_x               INTEGER,
    bbox_y               INTEGER,
    bbox_w               INTEGER,
    bbox_h               INTEGER,
    frame_snapshot_path  TEXT
);
CREATE INDEX IF NOT EXISTS idx_obj_location ON object_sightings (location_id);
CREATE INDEX IF NOT EXISTS idx_obj_detected ON object_sightings (detected_at);

-- 6. Seed 3 sample locations
INSERT INTO locations (id, name, type, address) VALUES
    ('LOC-001', 'Phoenix Mall',          'mall',    '123 Mall Road, City Centre'),
    ('LOC-002', 'City Campus',           'campus',  '456 University Ave, North District'),
    ('LOC-003', 'International Airport', 'airport', '789 Airport Blvd, West Terminal')
ON CONFLICT (id) DO NOTHING;
